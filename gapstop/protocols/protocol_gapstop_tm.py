# -*- coding: utf-8 -*-
# **************************************************************************
# *
# * Authors:     Scipion Team
# *
# * National Center of Biotechnology, CSIC, Spain
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# *
# **************************************************************************
import logging
from enum import Enum
from os.path import abspath, join
from typing import Union, List

import numpy as np

from gapstop import Plugin
from pwem.emlib.image import ImageHandler
from pwem.objects import VolumeMask, Volume
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.object import Pointer
from pyworkflow.protocol import STEPS_PARALLEL, PointerParam, FloatParam, StringParam, IntParam, GPU_LIST, \
    LEVEL_ADVANCED
from pyworkflow.utils import Message, makePath, getExt, createLink, cyanStr
from scipion.constants import PYTHON
from tomo.objects import SetOfCoordinates3D, SetOfTomograms, Tomogram, SetOfTiltSeries, CTFTomo
from tomo.utils import getObjFromRelation

logger = logging.getLogger(__name__)
# Inputs
IN_TOMOS = 'inTomos'
IN_CTF_SET = 'inCtfSet'
IN_TS_SET = 'inTsSet'
REF_VOL = 'refVol'
IN_MASK = 'inMask'
# Files and extensions
MRC = '.mrc'
TLT = '.tlt'


class gapStopOutputs(Enum):
    coordinates = SetOfCoordinates3D


class ProtGapStopTemplateMatching(EMProtocol):
    """GAPSTOPTM is able to leverage the power of GPU accelerated multi-node HPC systems to be efficiently
    used for template matching. It speeds up template matching by using an MPI-parallel layout and offloading
    the compute-heavy correlation kernel to one or more accelerator devices per MPI-process using jax.
    The template matching in GAPSTOPTM is algorithmically based on STOPGAP developed by W. Wan’s lab.
    """

    _label = 'template matching'
    _devStatus = BETA
    _possibleOutputs = gapStopOutputs
    stepsExecutionMode = STEPS_PARALLEL

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sRate = None
        self.tomoDict = None
        self.tsDict = None
        self.ctfDict = None
        self.ih = ImageHandler()

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam(IN_TOMOS, PointerParam,
                      pointerClass='SetOfTomograms',
                      important=True,
                      label='Tomograms')
        form.addParam(IN_CTF_SET, PointerParam,
                      pointerClass='SetOfCTFTomoSeries',
                      label="CTF tomo series",
                      allowsNull=True,
                      help='They are optional in case of the re-extraction of Relion particles.')
        form.addParam(IN_TS_SET, PointerParam,
                      pointerClass='SetOfTiltSeries',
                      allowsNull=True,
                      expertLevel=LEVEL_ADVANCED,
                      label='Tilt-series (opt.)',
                      help='Used to get the tilt angles. If empty, the protocol will try to reach, via relations,'
                           ' the tilt-series associated to the introduced tomograms.')
        # form.addParam(REF_VOL, PointerParam,
        #               pointerClass='Volume',
        #               important=True,
        #               label="Reference volume")
        # form.addParam(IN_MASK, PointerParam,
        #               pointerClass=VolumeMask,
        #               important=True,
        #               label='Reference mask')
        form.addParam('currentBin', IntParam,
                      allowsNull=False,
                      important=True,
                      label='Tomogram current binning factor',
                      help='Used to get the tomogram unbinned dimensions and sampling rate during the processing.')
        form.addParam('nTiles', IntParam,
                      default=1,
                      label='No. tiles to decompose the tomogram')
        group = form.addGroup('Angular sampling')
        group.addParam('coneAngle', FloatParam,
                       default=360,
                       label='Cone angle (deg.)')
        group.addParam('coneSampling', FloatParam,
                       default=10,
                       label='Cone sampling (deg.)')
        group.addParam('rotSymDeg', IntParam,
                       default='1',
                       label='Degree of rotational symmetry',
                       help='From 1, 2, ... to N, representing symmetries C1, C2, ... to CN, respectively. In case '
                            'of non-rotational symmetry, set it ti 1 (default).')
        form.addHidden(GPU_LIST, StringParam,
                       default='0',
                       label="Choose GPU IDs")
        form.addParallelSection(threads=1, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._initialize()
        pAngId = self._insertFunctionStep(self.prepareAnglesStep,
                                          prerequisites=[],
                                          needsGPU=False)
        for tsId in self.tomoDict.keys():
            cInputId = self._insertFunctionStep(self.convertInputStep, tsId,
                                                prerequisites=pAngId,
                                                needsGPU=False)

    # -------------------------- STEPS functions ------------------------------
    def _initialize(self):
        self.sRate = self._getFormAttrib(IN_TOMOS).getSamplingRate()
        tsSet = self._getFormAttrib(IN_TS_SET)
        tsSet = tsSet if tsSet else self._getTsFromRelations()
        tomoSet = self._getFormAttrib(IN_TOMOS)
        ctfSet = self._getFormAttrib(IN_CTF_SET)

        # Compute matching TS id among coordinates, the tilt-series and the CTFs, they all could be a subset
        tomosTsIds = set(tomoSet.getTSIds())
        tsIds = set(tsSet.getTSIds())
        ctfTsIds = set(ctfSet.getTSIds())
        presentTsIds = tomosTsIds & tsIds & ctfTsIds
        nonMatchingTsIds = (tomosTsIds ^ tsIds ^ ctfTsIds) - (tomosTsIds & tsIds & ctfTsIds)

        # Validate the intersection
        if len(presentTsIds) <= 0:
            raise Exception("There isn't any common tilt-series ids among the coordinates, CTFs, and tilt-series "
                            "introduced.")

        if len(nonMatchingTsIds) > 0:
            logger.info(cyanStr(f"TsIds not common in the introduced tomograms, CTFs, and "
                                f"tilt-series are:\n{presentTsIds}"))

        self.tomoDict = {tomo.getTsId(): tomo.clone() for tomo in tomoSet.iterItems()
                         if tomo.getTsId() in presentTsIds}
        self.tsDict = {ts.getTsId(): ts.clone() for ts in tsSet.iterItems()
                       if ts.getTsId() in presentTsIds}
        self.ctfDict = {ctf.getTsId(): ctf.clone(ignoreAttrs=[]) for ctf in ctfSet.iterItems()
                        if ctf.getTsId() in presentTsIds}

    def prepareAnglesStep(self):
        angleListFile = self._getCryoCatAngleFile()
        codePatch = f""" 
from cryocat import geom 
import numpy as np 

angles = geom.generate_angles({self.coneAngle.get()}, {self.coneSampling.get()}, symmetry={self.rotSymDeg.get()}) 
np.savetxt('{angleListFile}', angles, fmt='%.2f', delimiter=',') 
"""

        genAnglesPythonFile = self._getExtraPath('prepAngles.py')
        with open(genAnglesPythonFile, "w") as pyFile:
            pyFile.write(codePatch)
        Plugin.runGapStop(self, PYTHON, genAnglesPythonFile, isCryoCatExec=True)

    def convertInputStep(self, tsId: str):
        tomo = self.tomoDict[tsId]
        ts = self.tsDict[tsId]
        ctf = self.ctfDict[tsId]
        acq = ts.getAcquisition()
        tsDir = self._getCurrentTomoDir(tsId)
        makePath(tsDir)
        inTomoName = self._getWorkingTsIdFile(tsId, MRC)
        inTltName = self._getWorkingTsIdFile(tsId, TLT)
        self._convertOrLinkVolume(tomo, inTomoName)
        ts.generateTltFile(inTltName, includeDose=True)

        #  Defocus info:
        # "defocus1", "defocus2", "astigmatism", "phase_shift", "defocus_mean"
        nImgs = len(ctf)
        defocusData = np.zeros((nImgs, 5))
        for i, ctfTomo in enumerate(ctf.iterItems(orderBy=[CTFTomo.INDEX_FIELD], direction='ASC')):
            defocusData[i, 0] = ctfTomo.getDefocusU()
            defocusData[i, 1] = ctfTomo.getDefocusV()
            defocusData[i, 2] = ctfTomo.getDefocusAngle()
            defocusData[i, 4] = (ctfTomo.getDefocusU() + ctfTomo.getDefocusV()) / 2

        # Create the wedge list
        tltDoseData = np.loadtxt(inTltName)
        tltData = tltDoseData[:, 0]
        doseData = tltDoseData[:, 1]
        binfactor = self.currentBin.get()
        unBinnedtomoDims = np.array(tomo.getDimensions()) * binfactor
        unbinnedApix = self.sRate * binfactor

        codePatch = f"""
from cryocat import wedgeutils
import numpy as np

# Creates wedge list for single tomogram
tomo_dim = np.array({np.array2string(unBinnedtomoDims, separator=', ')})
tlt_data = np.array({np.array2string(tltData, separator=', ')})
ctf_data = np.array({np.array2string(defocusData, separator=', ')})
dose_data = np.array({np.array2string(doseData, separator=', ')})
wedgeutils.create_wedge_list_sg(
tomo_id='{tsId}',
tomo_dim=tomo_dim,
pixel_size={unbinnedApix},
tlt_file=tlt_data,
z_shift=0.0,
ctf_file=ctf_data,
dose_file=dose_data,
voltage={acq.getVoltage()},
amp_contrast={acq.getAmplitudeContrast()},
cs={acq.getSphericalAberration()},
output_file='{self._getCryoCatWedgesFiles(tsId)}',
drop_nan_columns=True
)
"""
        genWedgesPythonFile = join(self._getCurrentTomoDir(tsId), 'genWedgesList.py')
        with open(genWedgesPythonFile, "w") as pyFile:
            pyFile.write(codePatch)
        Plugin.runGapStop(self, PYTHON, genWedgesPythonFile, isCryoCatExec=True)

    # --------------------------- UTILS functions -----------------------------
    def _getFormAttrib(self, attribName: str, returnPointer: bool = False) -> Union[SetOfTiltSeries, Pointer]:
        inTsPointer = getattr(self, attribName)
        return inTsPointer if returnPointer else inTsPointer.get()

    def _getTsFromRelations(self) -> Union[SetOfTiltSeries, None]:
        inTomos = self._getFormAttrib(IN_TOMOS)
        return getObjFromRelation(inTomos, self, SetOfTiltSeries)

    def _getCryoCatAngleFile(self) -> str:
        return self._getExtraPath(f'angles_{self.coneSampling.get():.0f}_c{self.rotSymDeg.get()}.txt')

    def _getCryoCatWedgesFiles(self, tsId: str) -> str:
        return join(self._getCurrentTomoDir(tsId), 'wedges.star')

    def _getCurrentTomoDir(self, tsId: str) -> str:
        return self._getExtraPath(tsId)

    def _getWorkingTsIdFile(self, tsId: str, ext: str) -> str:
        return join(self._getCurrentTomoDir(tsId), tsId + ext)

    def _convertOrLinkVolume(self, inVolume: Volume, outVolume: str) -> None:
        """Converts a volume into a compatible MRC file or links it if already compatible"""
        inFn = inVolume.getFileName()
        # If compatible with dynamo. Attention!! Assuming is not a stack of mrc volumes!!
        if getExt(inFn) == MRC:
            createLink(abspath(inFn), outVolume)
        else:
            self.ih.convert(inVolume, outVolume)

    # --------------------------- INFO functions ------------------------------
    def _validate(self) -> List[str]:
        valMsg = []
        tsSet = self._getFormAttrib(IN_TS_SET)
        tsSetRel = self._getTsFromRelations()
        if not tsSet and not tsSetRel:
            valMsg.append('Unable to find via relations the tilt-series corresponding to the '
                          'introduced tomograms. Please introduce them manually (advanced parameters).')
        return valMsg
