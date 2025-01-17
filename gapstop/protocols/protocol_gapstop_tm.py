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
import numpy as np
from emtable import Table
from gapstop import Plugin
from pwem.convert import transformations
from pwem.emlib.image import ImageHandler
from pwem.objects import VolumeMask, Volume
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.object import Pointer, Set
from pyworkflow.protocol import STEPS_PARALLEL, PointerParam, FloatParam, StringParam, IntParam, GPU_LIST
from pyworkflow.utils import Message, makePath, getExt, createLink, cyanStr, removeBaseExt
from scipion.constants import PYTHON
from tomo.constants import BOTTOM_LEFT_CORNER
from tomo.objects import SetOfCoordinates3D, SetOfTomograms, SetOfTiltSeries, CTFTomo, SetOfCTFTomoSeries, Coordinate3D, \
    Tomogram
from tomo.utils import getObjFromRelation

logger = logging.getLogger(__name__)
# Inputs
IN_TOMOS = 'inTomos'
IN_CTF_SET = 'inCtfSet'
IN_TS_SET = 'inTsSet'
REF_VOL = 'reference'
IN_MASK = 'mask'
# Files and extensions
MRC = '.mrc'
EM = '.em'
TLT = '.tlt'
NPY = '.npy'
# tm_param.star columns
ROOTDIR = "rootdir"
OUTPUTDIR = "outputdir"
VOL_EXT = "vol_ext"
TOMO_NAME = "tomo_name"
TOMO_NUM = "tomo_num"
WEDGELIST_NAME = "wedgelist_name"
TMPL_NAME = "tmpl_name"
MASK_NAME = "mask_name"
SYMMETRY = "symmetry"
ANGLIST_ORDER = "anglist_order"
ANGLIST_NAME = "anglist_name"
SMAP_NAME = "smap_name"
OMAP_NAME = "omap_name"
LP_RAD = "lp_rad"
HP_RAD = "hp_rad"
BINNING = "binning"
TILING = "tiling"
#  Other params
RESULTS_DIR = 'results'
SCORES = 'scores'
ANGLES = 'angles'
PARTICLE_LIST_FILE = 'particleList'
PARTICLES_DATA_FILE = 'particlesData'

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
    program = 'gapstop'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sRate = None
        self.tomoDict = None
        self.tsDict = None
        self.ctfDict = None
        self.refName = None
        self.maskName = None
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
                      important=True,
                      allowsNull=True)
        form.addParam(IN_TS_SET, PointerParam,
                      pointerClass='SetOfTiltSeries',
                      allowsNull=True,
                      important=True,
                      label='Tilt-series (opt.)',
                      help='Used to get the tilt angles. If empty, the protocol will try to reach, via relations, '
                           'the tilt-series associated to the introduced tomograms.')
        form.addParam(REF_VOL, PointerParam,
                      pointerClass='Volume, SubTomogram',
                      important=True,
                      label="Reference volume")
        form.addParam(IN_MASK, PointerParam,
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
                      allowsNull=False,
                      important=True,
                      label='Tomogram current binning factor',
                      help='Used to get the tomogram unbinned dimensions and sampling rate during the processing.')
        form.addParam('partDiameter', FloatParam,
                      default=20,
                      label='Particle diameter',
                      help='Diameter of the particle to be used for extraction and clustering.')
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
        group = form.addGroup('Template filtering')
        group.addParam('lowPassFilter', FloatParam,
                       default=20,
                       label='Low-pass filter radius in Fourier px.',
                       help='To compute this value from the desired resolution, use following formula: '
                            'round(template_box_size * pixel_size / resolution) where template_box_size '
                            'is one dimension of the template.')
        group.addParam('highPassFilter', FloatParam,
                       default=1,
                       label='High-pass filter radius in Fourier px.',
                       help='In most cases the optimal value is 1 (i.e. no high-pass filter). To compute '
                            'this value from the desired resolution, use following formula: '
                            'round(template_box_size * pixel_size / resolution) where template_box_size '
                            'is one dimension of the template.')
        form.addHidden(GPU_LIST, StringParam,
                       default='0',
                       label="Choose GPU IDs")
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
            peId = self._insertFunctionStep(self.extractCoordinatesStep, tsId,
                                            prerequisites=tmId,
                                            needsGPU=False)
            cOutId = self._insertFunctionStep(self.createOutputStep, tsId,
                                              prerequisites=peId,
                                              needsGPU=False)
            closeSetStepDeps.append(cOutId)
        self._insertFunctionStep(self._closeOutputSet,
                                 prerequisites=closeSetStepDeps,
                                 needsGPU=False)

    # -------------------------- STEPS functions ------------------------------
    def _initialize(self):
        self.sRate = self._getFormAttrib(IN_TOMOS).getSamplingRate()
        tsSet = self._getFormAttrib(IN_TS_SET)
        tsSet = tsSet if tsSet else self._getTsFromRelations()
        tomoSet = self._getFormAttrib(IN_TOMOS)
        ctfSet = self._getFormAttrib(IN_CTF_SET)
        self.refName = self._genConvertedOrLinkedRefName(REF_VOL)
        self.maskName = self._genConvertedOrLinkedRefName(IN_MASK)

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

    def templateMatchingStep(self, tsId: str):
        logger.info(cyanStr(f'===> tsId = {tsId}: performing the template matching...'))
        args = 'run_tm '
        args += f'-n {self.nTiles.get()} '
        args += f'{self._genTmParamFileName(tsId)}'
        Plugin.runGapStop(self, self.program, args)

    def extractCoordinatesStep(self, tsId: str):
        tomo = self.tomoDict[tsId]
        tomoObjId = tomo.getObjId()
        scoresMapFile = self._getResultsFile(tsId, f'{SCORES}_0_{tomoObjId}')
        anglesMapFile = self._getResultsFile(tsId, f'{ANGLES}_0_{tomoObjId}')
        outParticleListFile = self._getResultsFile(tsId, PARTICLE_LIST_FILE)
        # TODO: Beware the scores threshold
        codePatch = f"""
from cryocat import tmana        

tmana.scores_extract_particles(
scores_map='{scoresMapFile}',
angles_map='{anglesMapFile}',
angles_list='{self._getCryoCatAngleFile()}',
tomo_id={tomo.getObjId()},
particle_diameter={self.partDiameter.get()},
object_id=None,
scores_threshold=0.12,
sigma_threshold=None,
cluster_size=None,
n_particles=None,
output_path='{outParticleListFile}',
output_type='emmotl',
angles_order='zxz',
symmetry='{self._getSymmetry()}',
angles_numbering=0)
"""
        extractCoordsPythonFile = join(self._getCurrentTomoDir(tsId), 'extractCoords.py')
        with open(extractCoordsPythonFile, "w") as pyFile:
            pyFile.write(codePatch)
        Plugin.runGapStop(self, PYTHON, extractCoordsPythonFile, isCryoCatExec=True)

    def createOutputStep(self, tsId: str):
        with self._lock:
            tomo = self.tomoDict[tsId]
            outCoords = self.createOutputSet()
            # The particle list is encoded in a .em file. Thus, it will be loaded and decoded.
            particlesDataNpyFile = self._loadParticleList(tsId)
            particlesData = np.load(particlesDataNpyFile)[0, :, :]  # Dims are [1, N, M] -> remove the first dim -> 2D
            for row in particlesData:
                coord = self._createCoordFromRow(tomo, row)
                outCoords.append(coord)
            outCoords.write()
            self._store(outCoords)

    # --------------------------- UTILS functions -----------------------------
    def createOutputSet(self):
        outCoords = getattr(self, self._possibleOutputs.coordinates.name, None)
        if outCoords:
            outCoords.enableAppend()
        else:
            inTomosPointer = self._getFormAttrib(IN_TOMOS, returnPointer=True)
            inTomos = inTomosPointer.get()
            outCoords = SetOfCoordinates3D.create(self._getPath(), template="coordinates%s")
            outCoords.setPrecedents(inTomos)
            outCoords.setSamplingRate(inTomos.getSamplingRate())
            outCoords.setBoxSize(self._getFormAttrib(REF_VOL).getDim()[0])
            outCoords.setStreamState(Set.STREAM_OPEN)

            self._defineOutputs(**{self._possibleOutputs.coordinates.name: outCoords})
            self._defineSourceRelation(inTomosPointer, outCoords)

        return outCoords

    def _createCoordFromRow(self, tomo: Tomogram, row: np.array):
        """The particle list is encoded in a .em file. Thus, it will be loaded and decoded. The data represented
        on each column is explained in https://cryocat.readthedocs.io/latest/user_guide/motl_basics.html:
        Columns:
        0 - "score” - a quality metric (typically cross-correlation value between the particle and the reference)
        1 - "geom1” - a free geometric property
        2 - "geom2” - a free geometric property
        3 - "subtomo_id” - a subtomogram id; IMPORTANT many functions rely on this one to be unique
        4 - "tomo_id” - a tomogram id to which the particle is affiliated to
        5 - "object_id” - an object id to which the particle is affiliated to
        6 - "subtomo_mean” - a mean value of the subtomogram
        7 - "x” - a position in the tomogram (an integer value), typically used for subtomogram extraction
        8 - “y” - a position in the tomogram (an integer value), typically used for subtomogram extraction
        9 - “z” - a position in the tomogram (an integer value), typically used for subtomogram extraction
        10 - “shift_x” - shift of the particle in X direction (a float value); to complete position of a particle is given by x + shift_x
        11 - “shift_y” - shift of the particle in Y direction (a float value); to complete position of a particle is given by y + shift_y
        12 - “shift_z” - shift of the particle in Z direction (a float value); to complete position of a particle is given by z + shift_z
        13 - “geom3” - a free geometric property
        14 - “geom4” - a free geometric property
        15 - “geom5” - a free geometric property
        16 - “phi” - a phi angle describing rotation around the first Z axis (following Euler zxz convention)
        17 - “psi” - a psi angle describing rotation around the second Z axis (following Euler zxz convention)
        18 - “theta” - a thetha angle describing rotation around the X axis (following Euler zxz convention)
        19 - “class” - a class of the particle"""
        coord = Coordinate3D()
        score = row[0]
        x, y, z = row[7:10]
        sx, sy, sz = row[10:13]
        phi, psi, theta = row[16:19]
        matrix = self.eulerAngles2matrix(phi, theta, psi, sx, sy, sz)
        coord.setScore(score)
        coord.setVolume(tomo)
        coord.setPosition(x, y, z, BOTTOM_LEFT_CORNER)
        coord.setMatrix(matrix)
        return coord

    @staticmethod
    def eulerAngles2matrix(tdrot, tilt, narot, shiftx, shifty, shiftz):
        # Relevant info:
        #   * GapStop's transformation system is ZXZ
        #   * Sscipion = R * (-Sgapstop) ==> Sgapstop = Rinv * (-Sscipion)
        M = np.eye(4)
        sx = float(shiftx)
        sy = float(shifty)
        sz = float(shiftz)
        Sgapstop = np.array([sx, sy, sz])
        tdrot = np.deg2rad(float(tdrot))
        narot = np.deg2rad(float(narot))
        tilt = np.deg2rad(float(tilt))
        R = transformations.euler_matrix(tdrot, tilt, narot, axes='szxz')
        R = R[:3, :3]
        Sscipion = - np.dot(R, Sgapstop)
        M[:3, :3] = R
        M[:3, 3] = Sscipion
        return M

    def _getFormAttrib(self, attribName: str, returnPointer: bool = False) -> Union[SetOfTiltSeries,
    SetOfTomograms, SetOfCTFTomoSeries, Volume, Pointer, None]:
        inTsPointer = getattr(self, attribName, None)
        if not inTsPointer:
            return None
        else:
            return inTsPointer if returnPointer else inTsPointer.get()

    def _getTsFromRelations(self) -> Union[SetOfTiltSeries, None]:
        inTomos = self._getFormAttrib(IN_TOMOS)
        return getObjFromRelation(inTomos, self, SetOfTiltSeries)

    def _getCryoCatAngleFile(self) -> str:
        return self._getExtraPath(f'angles_{self.coneSampling.get():.0f}_{self._getSymmetry()}.txt')

    def _getCryoCatWedgesFiles(self, tsId: str) -> str:
        return join(self._getCurrentTomoDir(tsId), 'wedges.star')

    def _getCurrentTomoDir(self, tsId: str) -> str:
        return self._getExtraPath(tsId)

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

    def _getResultsFile(self, tsId: str, fileName: str, ext: str=EM) -> str:
        return join(self._getCurrentTomoDir(tsId), RESULTS_DIR, fileName + ext)

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
                EM, # vol_ext
                basename(self._getWorkingTsIdFile(tsId, MRC)),  # tomo_name
                tomoObjId, # tomo_num,
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
                self.currentBin.get(),  # tomogram_binning
                'new'  # tiling
            ]
            paramTable.addRow(*paramList)
            paramTable.writeStar(f, tableName=tsId)

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

    def _loadParticleList(self, tsId: str) -> str:
        """The particle list is encoded in a .em file. Thus, it will be loaded and decoded. The data represented
        on each column is explained in https://cryocat.readthedocs.io/latest/user_guide/motl_basics.html"""
        particlesEmFile = self._getResultsFile(tsId, PARTICLE_LIST_FILE)
        outParticlesNpyFile = self._getResultsFile(tsId, PARTICLES_DATA_FILE, NPY)
        codePatch = f""" 
import emfile 
import numpy as np

_, data = emfile.read('{particlesEmFile}')
np.save('{outParticlesNpyFile}', data)
        """
        readEmParticlePythonFile = self._getExtraPath('readEmParticles.py')
        with open(readEmParticlePythonFile, "w") as pyFile:
            pyFile.write(codePatch)
        Plugin.runGapStop(self, PYTHON, readEmParticlePythonFile, isCryoCatExec=True)
        return outParticlesNpyFile

    # --------------------------- INFO functions ------------------------------
    def _validate(self) -> List[str]:
        valMsg = []
        tsSet = self._getFormAttrib(IN_TS_SET)
        tsSetRel = self._getTsFromRelations()
        if not tsSet and not tsSetRel:
            valMsg.append('Unable to find via relations the tilt-series corresponding to the '
                          'introduced tomograms. Please introduce them manually (advanced parameters).')
        return valMsg
