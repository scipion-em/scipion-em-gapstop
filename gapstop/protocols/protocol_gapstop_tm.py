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
from os.path import abspath, join, basename
from typing import Union, List
import mrcfile
import numpy as np
from emtable import Table
from gapstop import Plugin
from gapstop.constants import *
from gapstop.objects import SetOfGapStopScoreTomograms, GapStopScoreTomogram
from gapstop.protocols.protocol_base import ProtGapStopBase
from pwem.emlib.image import ImageHandler
from pwem.objects import VolumeMask, Volume
from pyworkflow import BETA
from pyworkflow.object import Set, String
from pyworkflow.protocol import STEPS_PARALLEL, PointerParam, FloatParam, StringParam, IntParam, GPU_LIST, BooleanParam, \
    LEVEL_ADVANCED
from pyworkflow.utils import Message, makePath, getExt, createLink, cyanStr
from scipion.constants import PYTHON
from tomo.objects import SetOfTiltSeries, CTFTomo
from tomo.utils import getObjFromRelation

logger = logging.getLogger(__name__)


class GapStopTMOutputs(Enum):
    scoreTomogrmas = SetOfGapStopScoreTomograms


class ProtGapStopTemplateMatching(ProtGapStopBase):
    """GAPSTOPTM is able to leverage the power of GPU accelerated multi-node HPC systems to be efficiently
    used for template matching. It speeds up template matching by using an MPI-parallel layout and offloading
    the compute-heavy correlation kernel to one or more accelerator devices per MPI-process using jax.
    The template matching in GAPSTOPTM is algorithmically based on STOPGAP developed by W. Wan’s lab.
    """

    _label = 'template matching'
    _devStatus = BETA
    _possibleOutputs = GapStopTMOutputs
    stepsExecutionMode = STEPS_PARALLEL
    program = 'gapstop'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tomosSRate = None
        self.tomosBinning = None
        self.tomoDict = None
        self.tsDict = None
        self.ctfDict = None
        self.refName = None
        self.maskName = None
        self.ih = ImageHandler()
        self.failedTsIds = []
        self.failedTsIdsStr = String()

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
                      important=True,
                      allowsNull=True)
        form.addParam(IN_TS_SET, PointerParam,
                      pointerClass='SetOfTiltSeries',
                      allowsNull=True,
                      expertLevel=LEVEL_ADVANCED,
                      label='Tilt-series (opt.)',
                      help='Used to get the tilt angles. If empty, the protocol will try to reach, via relations, '
                           'the tilt-series associated to the introduced CTFs.')
        group = form.addGroup('Reference')
        group.addParam(REF_VOL, PointerParam,
                       pointerClass='Volume',
                       important=True,
                       label="Reference volume")
        group.addParam('doInvertRefContrast', BooleanParam,
                       default=True,
                       label='Invert contrast?',
                       important=True,
                       help='The contrast of the template has to be the same as of the tomogram. If the '
                            'tomogram has features in black (which is typically for cryoET) then the template '
                            'has to have the same representation. For example, Relion outputs inverted '
                            'contrast (features are white) and such maps have to be inverted prior running '
                            'the GapStop_TM.')
        group.addParam(IN_MASK, PointerParam,
                       pointerClass=VolumeMask,
                       important=True,
                       label='Reference mask',
                       help='It is used in two ways. First, the bounding box is computed to contain all values '
                            'equal to 1. This can reduce computation time since the bounding box can be way smaller '
                            'than the tomogram. The second time, the binary mask is used, is by outputing '
                            'scores map where the mask is used to multiply the scores map so only values '
                            'corresponding to the mask regions with value 1 are kept. For the latter, same '
                            'effect can be achieved by multiplying the scores map after the TM run.')
        form.addParam('currentBin', IntParam,
                      allowsNull=True,
                      expertLevel=LEVEL_ADVANCED,
                      label='Tomogram current binning factor (opt.)',
                      help='Used to get the tomogram unbinned dimensions and sampling rate during the processing. '
                           'If not set, it will be calculated considering the sampling rate of the tilt-series '
                           'associated to the introduced CTFs and the sampling rate of the introduced tomograms.')
        form.addParam('nTiles', IntParam,
                      default=1,
                      label='No. tiles to decompose the tomogram')
        form.addSection(label='Angular sampling')
        form.addParam('coneAngle', FloatParam,
                      default=360,
                      label='Cone angle (deg.)')
        form.addParam('coneSampling', FloatParam,
                      default=10,
                      label='Cone sampling (deg.)')
        form.addParam('rotSymDeg', IntParam,
                      default='1',
                      label='Degree of rotational symmetry',
                      help='From 1, 2, ... to N, representing symmetries C1, C2, ... to CN, respectively. In case '
                           'of non-rotational symmetry, set it ti 1 (default).')
        form.addSection(label='Template filtering')
        form.addParam('lowPassFilter', FloatParam,
                      default=20,
                      label='Low-pass filter radius in Fourier px.',
                      help='To compute this value from the desired resolution, use following formula: '
                           'round(template_box_size * pixel_size / resolution) where template_box_size '
                           'is one dimension of the template.')
        form.addParam('highPassFilter', FloatParam,
                      default=1,
                      label='High-pass filter radius in Fourier px.',
                      help='In most cases the optimal value is 1 (i.e. no high-pass filter). To compute '
                           'this value from the desired resolution, use following formula: '
                           'round(template_box_size * pixel_size / resolution) where template_box_size '
                           'is one dimension of the template.')
        form.addHidden(GPU_LIST, StringParam,
                       default='0',
                       label="Choose GPU IDs",
                       help='GPU device/s to be used.')
        form.addParallelSection(threads=1, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._initialize()
        closeSetStepDeps = []
        cRId = self._insertFunctionStep(self.convertReferenceStep,
                                        prerequisites=[],
                                        needsGPU=False)
        pAngId = self._insertFunctionStep(self.prepareAnglesStep,
                                          prerequisites=cRId,
                                          needsGPU=False)
        for tsId in self.tomoDict.keys():
            cInputId = self._insertFunctionStep(self.convertInputStep, tsId,
                                                prerequisites=pAngId,
                                                needsGPU=False)
            tmId = self._insertFunctionStep(self.templateMatchingStep, tsId,
                                            prerequisites=cInputId,
                                            needsGPU=True)
            cOutId = self._insertFunctionStep(self.createOutputStep, tsId,
                                              prerequisites=tmId,
                                              needsGPU=False)
            closeSetStepDeps.append(cOutId)
        self._insertFunctionStep(self.closeOutputSetStep,
                                 prerequisites=closeSetStepDeps,
                                 needsGPU=False)

    # -------------------------- STEPS functions ------------------------------
    def _initialize(self):
        tsSet = self._getTsSet()
        tomoSet = self._getFormAttrib(IN_TOMOS)
        ctfSet = self._getFormAttrib(IN_CTF_SET)
        self.refName = self._genConvertedOrLinkedRefName(REF_VOL)
        self.maskName = self._genConvertedOrLinkedRefName(IN_MASK)
        self.tomosSRate = tomoSet.getSamplingRate()
        self.tomosBinning = self._getTomogramsBinning()

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
                                f"tilt-series are: {nonMatchingTsIds}"))

        self.tomoDict = {tomo.getTsId(): tomo.clone() for tomo in tomoSet.iterItems()
                         if tomo.getTsId() in presentTsIds}
        self.tsDict = {ts.getTsId(): ts.clone() for ts in tsSet.iterItems()
                       if ts.getTsId() in presentTsIds}
        self.ctfDict = {ctf.getTsId(): ctf.clone(ignoreAttrs=[]) for ctf in ctfSet.iterItems()
                        if ctf.getTsId() in presentTsIds}

    def convertReferenceStep(self):
        # Convert or link the reference
        ref = self._getFormAttrib(REF_VOL)
        if self.doInvertRefContrast.get():
            self._invertReference(ref)
        else:
            self._convertOrLinkVolume(ref, self.refName)
        # Convert or link the mask
        mask = self._getFormAttrib(IN_MASK)
        self._convertOrLinkVolume(mask, self.maskName)

    def prepareAnglesStep(self):
        logger.info(cyanStr('Generating the file with Euler angles specifying the rotations...'))
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
        try:
            tomo = self.tomoDict[tsId]
            ts = self.tsDict[tsId]
            ctf = self.ctfDict[tsId]
            acq = ts.getAcquisition()
            tomoObjId = tomo.getObjId()
            tsDir = self._getCurrentTomoDir(tsId)
            makePath(tsDir)

            # Convert or link the current tomogram
            logger.info(cyanStr(f'tsId: {tsId}: converting or linking the tomogram...'))
            inTomoName = self._getWorkingTsIdFile(tsId, MRC)
            self._convertOrLinkVolume(tomo, inTomoName)

            #  Defocus info:
            # "defocus1", "defocus2", "astigmatism", "phase_shift", "defocus_mean"
            logger.info(cyanStr(f'tsId: {tsId}: generating the wedge list file...'))
            nImgs = len(ctf)
            defocusData = np.zeros((nImgs, 5))
            for i, ctfTomo in enumerate(ctf.iterItems(orderBy=[CTFTomo.INDEX_FIELD], direction='ASC')):
                defocusData[i, 0] = ctfTomo.getDefocusU()
                defocusData[i, 1] = ctfTomo.getDefocusV()
                defocusData[i, 2] = ctfTomo.getDefocusAngle()
                defocusData[i, 4] = (ctfTomo.getDefocusU() + ctfTomo.getDefocusV()) / 2

            # Tilt angles and dose
            inTltName = self._getWorkingTsIdFile(tsId, TLT)
            ts.generateTltFile(inTltName, includeDose=True)
            tltDoseData = np.loadtxt(inTltName)
            tltData = tltDoseData[:, 0]
            doseData = tltDoseData[:, 1]

            # Create the wedge list
            wedgesStarFile = self._getCryoCatWedgesFiles(tsId)
            binfactor = self.tomosBinning
            unBinnedtomoDims = np.array(tomo.getDimensions()) * binfactor
            unbinnedApix = self.tomosSRate * binfactor
            codePatch = f"""
from cryocat import wedgeutils
import numpy as np

# Creates wedge list for single tomogram
tomo_dim = np.array({np.array2string(unBinnedtomoDims, separator=', ')})
tlt_data = np.array({np.array2string(tltData, separator=', ')})
ctf_data = np.array({np.array2string(defocusData, separator=', ')})
dose_data = np.array({np.array2string(doseData, separator=', ')})
wedgeutils.create_wedge_list_sg(
tomo_id='{tomoObjId}',
tomo_dim=tomo_dim,
pixel_size={unbinnedApix:.3f},
tlt_file=tlt_data,
z_shift=0.0,
ctf_file=ctf_data,
dose_file=dose_data,
voltage={acq.getVoltage()},
amp_contrast={acq.getAmplitudeContrast()},
cs={acq.getSphericalAberration()},
output_file='{wedgesStarFile}',
drop_nan_columns=True)
"""
            genWedgesPythonFile = join(self._getCurrentTomoDir(tsId), 'genWedgesList.py')
            with open(genWedgesPythonFile, "w") as pyFile:
                pyFile.write(codePatch)
            Plugin.runGapStop(self, PYTHON, genWedgesPythonFile, isCryoCatExec=True)
            self._fixWedgesFile(wedgesStarFile)

            # Generate the tm_params.star
            logger.info(cyanStr(f'tsId: {tsId}: generating the tm_params.star file...'))
            self._createTmParamsFile(tsId, tomoObjId)
        except:
            self.failedTsIds.append(tsId)

    def templateMatchingStep(self, tsId: str):
        if tsId not in self.failedTsIds:
            try:
                logger.info(cyanStr(f'===> tsId = {tsId}: performing the template matching...'))
                args = 'run_tm '
                args += f'-n {self.nTiles.get()} '
                args += f'{self._genTmParamFileName(tsId)}'
                Plugin.runGapStop(self, self.program, args)
            except:
                self.failedTsIds.append(tsId)

    def createOutputStep(self, tsId: str):
        if tsId not in self.failedTsIds:
            with self._lock:
                tomo = self.tomoDict[tsId]
                convertedOrLinkedTomoFile = self._getWorkingTsIdFile(tsId, MRC)
                tomoNum = tomo.getObjId()
                scoreTomoSet = self.createOutputSet()
                # Create the corresponding scoreTomo
                scoreTomo = GapStopScoreTomogram()
                scoresMap = self._getResultsFile(tsId, self._getResultsBName(SCORES, tomoNum))
                anglesMap = self._getResultsFile(tsId, self._getResultsBName(ANGLES, tomoNum))
                anglesList = self._getCryoCatAngleFile()
                scoreTomo.setTsId(tsId)
                scoreTomo.setFileName(scoresMap)
                scoreTomo.setTomoFile(convertedOrLinkedTomoFile)
                scoreTomo.setAnglesMap(anglesMap)
                scoreTomo.setAnglesList(anglesList)
                scoreTomo.setTomoNum(tomoNum)
                scoreTomo.setSymmetry(self._getSymmetry())
                # Append to the set and store
                scoreTomoSet.append(scoreTomo)
                scoreTomoSet.write()
                self._store(scoreTomoSet)

    def closeOutputSetStep(self):
        scoreTomoSet = getattr(self, self._possibleOutputs.scoreTomogrmas.name, None)
        if scoreTomoSet:
            self._closeOutputSet()
        else:
            raise Exception('No gapStopTM scored tomograms were generated. Maybe the tomograms are too large '
                            'for the GPU/s used. Consider to bin them before and/or introduce a higher number in '
                            'the parameter "No. tiles to descompose the tomogram".')
        if self.failedTsIds:
            self.failedTsIdsStr.set(str(self.failedTsIds))
            self._store(self.failedTsIdsStr)

    # --------------------------- UTILS functions -----------------------------
    def createOutputSet(self):
        scoreTomoSet = getattr(self, self._possibleOutputs.scoreTomogrmas.name, None)
        if scoreTomoSet:
            scoreTomoSet.enableAppend()
        else:
            inTomosPointer = self._getFormAttrib(IN_TOMOS, returnPointer=True)
            inTomos = inTomosPointer.get()
            scoreTomoSet = SetOfGapStopScoreTomograms.create(self._getPath(), template="scoreTomograms%s")
            scoreTomoSet.setSamplingRate(inTomos.getSamplingRate())
            scoreTomoSet.setStreamState(Set.STREAM_OPEN)

            self._defineOutputs(**{self._possibleOutputs.scoreTomogrmas.name: scoreTomoSet})
            self._defineSourceRelation(inTomosPointer, scoreTomoSet)

        return scoreTomoSet

    def _getTsSet(self) -> SetOfTiltSeries:
        tsSet = self._getFormAttrib(IN_TS_SET)
        return tsSet if tsSet else self._getTsFromRelations()

    def _getTomogramsBinning(self) -> int:
        formBin = self.currentBin.get()
        return formBin if formBin else self._calculateTomogramsBinning()

    def _calculateTomogramsBinning(self) -> int:
        tsSetSRate = self._getTsSet().getSamplingRate()
        tomoSRate = self.tomosSRate
        tomosBinning = round(tomoSRate / tsSetSRate)
        logger.info(cyanStr(f"Tomogrmas binning calculated -> binning = {tomosBinning}"))
        return tomosBinning

    def _getTsFromRelations(self) -> Union[SetOfTiltSeries, None]:
        inTomos = self._getFormAttrib(IN_CTF_SET)
        return getObjFromRelation(inTomos, self, SetOfTiltSeries)

    def _getCryoCatAngleFile(self) -> str:
        return self._getExtraPath(f'angles_{self.coneSampling.get():.0f}_{self._getSymmetry()}.txt')

    def _getCryoCatWedgesFiles(self, tsId: str) -> str:
        return join(self._getCurrentTomoDir(tsId), 'wedges.star')

    def _getWorkingTsIdFile(self, tsId: str, ext: str) -> str:
        return join(self._getCurrentTomoDir(tsId), tsId + ext)

    def _genConvertedOrLinkedRefName(self, baseName: str) -> str:
        return self._getExtraPath(baseName + MRC)

    def _convertOrLinkVolume(self, inVolume: Volume, outVolume: str) -> None:
        """Converts a volume into a compatible MRC file or links it if already compatible"""
        inFn = inVolume.getFileName()
        # If compatible with gapstop. Attention!! Assuming is not a stack of mrc volumes!!
        if getExt(inFn) == MRC:
            createLink(abspath(inFn), outVolume)
        else:
            self.ih.convert(inVolume, outVolume)

    def _genTmParamFileName(self, tsId: str) -> str:
        return join(self._getCurrentTomoDir(tsId), 'tm_params.star')

    def _getResultsFile(self, tsId: str, fileName: str, ext: str = MRC) -> str:
        return join(self._getCurrentTomoDir(tsId), RESULTS_DIR, fileName + ext)

    @staticmethod
    def _getResultsBName(fileName: str, tomoNum: int):
        """The resulting scores and angles files are named NAME_0_TomoNum."""
        return f'{fileName}_0_{tomoNum}'

    def _getSymmetry(self):
        return f'C{self.rotSymDeg.get()}'

    @staticmethod
    def _fixWedgesFile(wedgesStarFile):
        """There is a bug in the wedges star file generation that introduces a blank line between
        the column names and the contents. This causes the template matching program to detect it
        as an empty star file. Therefore, we will remove the blank lines from that file."""
        # Read the wedges file
        with open(wedgesStarFile, 'r') as file:
            lineas = file.readlines()

        # Remove the blank lines
        lineas = [linea for linea in lineas if linea.strip() != '']

        # Overwrite the file with the fixed contents
        with open(wedgesStarFile, 'w') as file:
            file.writelines(lineas)

    def _createTmParamsFile(self, tsId: str, tomoObjId: int):
        """See param explanation here -->
        https://gitlab.mpcdf.mpg.de/bturo/gapstop_tm/-/blob/main/doc/user_manual/tm_params.rst?ref_type=heads"""
        paramTable = Table(columns=self._getTmParamStarFields())
        with open(self._genTmParamFileName(tsId), 'w') as f:
            paramList = [
                self._getCurrentTomoDir(tsId) + '/',  # rootdir
                RESULTS_DIR + '/',  # outputdir
                MRC,  # vol_ext
                basename(self._getWorkingTsIdFile(tsId, MRC)),  # tomo_name
                tomoObjId,  # tomo_num,
                basename(self._getCryoCatWedgesFiles(tsId)),  # wedgelist_name
                join('..', basename(self.refName)),  # template_name
                join('..', basename(self.maskName)),  # tomo_mask_name
                f'{self._getSymmetry()}',  # symmetry
                'zxz',  # angslist_order
                join('..', basename(self._getCryoCatAngleFile())),  # anglist_name
                SCORES,  # smap_name
                ANGLES,  # omap_name
                self.lowPassFilter.get(),  # lp_rad
                self.highPassFilter.get(),  # hp_rad
                self.tomosBinning,  # tomogram_binning
                'new'  # tiling
            ]
            paramTable.addRow(*paramList)
            paramTable.writeStar(f, tableName=tsId)

    def _invertReference(self, ref: Volume):
        logger.info(cyanStr("Inverting the reference contrast"))
        with mrcfile.open(ref.getFileName()) as origRef:
            origData = origRef.data
            with mrcfile.new(self.refName) as invertedRef:
                invertedRef.set_data(-1 * origData)
                invertedRef.voxel_size = ref.getSamplingRate()

    @staticmethod
    def _getTmParamStarFields():
        return [
            ROOTDIR,
            OUTPUTDIR,
            VOL_EXT,
            TOMO_NAME,
            TOMO_NUM,
            WEDGELIST_NAME,
            TMPL_NAME,
            MASK_NAME,
            SYMMETRY,
            ANGLIST_ORDER,
            ANGLIST_NAME,
            SMAP_NAME,
            OMAP_NAME,
            LP_RAD,
            HP_RAD,
            BINNING,
            TILING]

    # --------------------------- INFO functions ------------------------------
    def _validate(self) -> List[str]:
        valMsg = []
        tsSet = self._getFormAttrib(IN_TS_SET)
        tsSetRel = self._getTsFromRelations()
        if not tsSet:
            if not tsSetRel or not self.currentBin.get():
                valMsg.append('Unable to find via relations the tilt-series corresponding to the '
                              'introduced tomograms. Please introduce them manually (advanced parameters).')
        return valMsg

    def _summary(self) -> List[str]:
        msg = []
        if self.isFinished():
            msg.append('*GapStop_TM is composed of 2 steps*. To extract the coordinates from the scored '
                       'tomograms calculated, call the protocol *gapstop - extract coordinates*.')
            failedStrs = self.failedTsIdsStr.get()
            if failedStrs:
                msg.append(f'The following tsIds were not possible to be processed: *{failedStrs}*')
        return msg

