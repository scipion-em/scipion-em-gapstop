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
from enum import Enum
from os.path import join

import mrcfile
import numpy as np

from gapstop import Plugin
from gapstop.constants import *
from gapstop.protocols.protocol_base import ProtGapStopBase
from pwem.convert import transformations
from pyworkflow import BETA
from pyworkflow.object import Set
from pyworkflow.protocol import PointerParam, IntParam, FloatParam, GT, STEPS_PARALLEL, LE, GE
from pyworkflow.utils import Message, makePath
from scipion.constants import PYTHON
from tomo.constants import BOTTOM_LEFT_CORNER
from tomo.objects import SetOfCoordinates3D, Tomogram, Coordinate3D, SetOfTomograms
from tomo.utils import getObjFromRelation


class GSExtractCoordsOutputs(Enum):
    coordinates = SetOfCoordinates3D


class ProtGapStopExtractCoords(ProtGapStopBase):
    """Extracts coordinates from score maps produced by template matching with GAPSTOP(TM)."""

    _label = 'extract coordinates'
    _devStatus = BETA
    _possibleOutputs = GSExtractCoordsOutputs
    stepsExecutionMode = STEPS_PARALLEL

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scoreTomoDict = None
        self.tomoSet = None

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam(IN_SCORE_TOMOS, PointerParam,
                      pointerClass='SetOfGapStopScoreTomograms',
                      important=True,
                      label='Score tomograms',
                      help='They are the result of the gapstop template matching.')
        form.addParam('partDiameter', FloatParam,
                      default=10,
                      important=True,
                      validators=[GT(0)],
                      label='Particle diameter (px)',
                      help='Diameter of the particle to be used for extraction and clustering.\n\n'
                           'In this context, it has more correspondence with a mathematical '
                           'representation than with the exact physical diameter of the particle. '
                           'It is a way to establish a distance criterion for the search and clustering '
                           'of points in the correlation map. In a tomogram, particles may not be '
                           'perfectly spherical, and their diameters may vary. By using a value for '
                           'the "particle diameter," you are defining a distance threshold that is '
                           'useful for identifying and grouping points that are likely to belong to '
                           'the same particle or structure in the score map.\n\n'
                           'In some contexts, the density of points in the score map may vary. '
                           'A smaller diameter can reduce noise by focusing on more compact groups, '
                           'while in other cases, a larger diameter may be necessary to cluster '
                           'larger and more spaced-out particles.')
        form.addParam('percentile', FloatParam,
                      default=99.5,
                      important=True,
                      validators=[GE(0), LE(100)],
                      label='Percentile value',
                      help='Percentile value [in range [0, 100] of the GapStopScoreTomograms that will be used '
                           'to determine the best candidates to be particles. If a max. number of coordinates '
                           'per tomogram is provided, they will be the best N decending sorted by score. Normally, '
                           'the recommended values are from 99.5 to 99.99.')
        form.addParam('numberOfCoords', IntParam,
                      default=-1,
                      label='Max no. coordinates per tomogram',
                      help='If set to -1, all the coordinates resulting after having applied the particle diameter '
                           'and the score threshold will be saved. Any other case, the first N coordinates, sorted '
                           'by score, will be saved.')
        form.addParallelSection(threads=1, mpi=0)

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._initialize()
        closeSetStepDeps = []
        for tsId in self.scoreTomoDict:
            ecId = self._insertFunctionStep(self.extractCoordinatesStep, tsId,
                                            prerequisites=[],
                                            needsGPU=False)
            cOutId = self._insertFunctionStep(self.createOutputStep, tsId,
                                              prerequisites=ecId,
                                              needsGPU=False)
            closeSetStepDeps.append(cOutId)
        self._insertFunctionStep(self._closeOutputSet,
                                 prerequisites=closeSetStepDeps,
                                 needsGPU=False)

    # -------------------------- STEPS functions ------------------------------
    def _initialize(self):
        scoreTomoSet = self._getFormAttrib(IN_SCORE_TOMOS)
        self.scoreTomoDict = {sTomo.getTsId(): sTomo.clone() for sTomo in scoreTomoSet.iterItems()}
        self.tomoSet = self._getTomosFromRelations()

    def extractCoordinatesStep(self, tsId: str):
        sTomo = self.scoreTomoDict[tsId]
        tsIdExtraDir = self._getCurrentTomoDir(tsId)
        makePath(tsIdExtraDir)
        outParticleListFile = self._getTsIdExtraDirFile(tsId, PARTICLE_LIST_FILE)
        scoresMap = sTomo.getFileName()
        # scores_threshold = {self.scoresThreshold.get()},
        codePatch = f"""
from cryocat import tmana        

tmana.scores_extract_particles(
scores_map='{scoresMap}',
angles_map='{sTomo.getAnglesMap()}',
angles_list='{sTomo.getAngleList()}',
tomo_id={sTomo.getTomoNum()},
particle_diameter={self.partDiameter.get()},
object_id=None,
scores_threshold={self._getScorePercentileValue(scoresMap)},
sigma_threshold=None,
cluster_size=None,
n_particles=None,
output_path='{outParticleListFile}',
output_type='emmotl',
angles_order='zxz',
symmetry='{sTomo.getSymmetry()}',
angles_numbering=0)
"""
        extractCoordsPythonFile = join(tsIdExtraDir, 'extractCoords.py')
        with open(extractCoordsPythonFile, "w") as pyFile:
            pyFile.write(codePatch)
        Plugin.runGapStop(self, PYTHON, extractCoordsPythonFile, isCryoCatExec=True)

    def createOutputStep(self, tsId: str):
        with self._lock:
            outCoords = self.createOutputSet()
            # The particle list is encoded in a .em file. Thus, it will be loaded and decoded.
            particlesDataNpyFile = self._loadParticleList(tsId)
            particlesData = np.load(particlesDataNpyFile)
            tomo = self.tomoSet.getItem(Tomogram.TS_ID_FIELD, tsId)
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
            inScoreTomosPointer = self._getFormAttrib(IN_SCORE_TOMOS, returnPointer=True)
            inScoreTomos = inScoreTomosPointer.get()
            outCoords = SetOfCoordinates3D.create(self._getPath(), template="coordinates%s")
            outCoords.setPrecedents(self.tomoSet)
            outCoords.setSamplingRate(inScoreTomos.getSamplingRate())
            outCoords.setBoxSize(self.partDiameter.get())
            outCoords.setStreamState(Set.STREAM_OPEN)

            self._defineOutputs(**{self._possibleOutputs.coordinates.name: outCoords})
            self._defineSourceRelation(inScoreTomosPointer, outCoords)

        return outCoords

    def _getTomosFromRelations(self) -> SetOfTomograms:
        inScoreTomos = self._getFormAttrib(IN_SCORE_TOMOS)
        return getObjFromRelation(inScoreTomos, self, SetOfTomograms)

    def _getScorePercentileValue(self, tomoFileName: str) -> float:
        with mrcfile.mmap(tomoFileName) as mrc:
            data = mrc.data
            percentileVal = np.percentile(data, self.percentile.get())
            return percentileVal


    def _loadParticleList(self, tsId: str) -> str:
        """The particle list is encoded in a .em file. Thus, it will be loaded and decoded. The data represented
        on each column is explained in https://cryocat.readthedocs.io/latest/user_guide/motl_basics.html"""
        particlesEmFile = self._getTsIdExtraDirFile(tsId, PARTICLE_LIST_FILE)
        outParticlesNpyFile = self._getTsIdExtraDirFile(tsId, PARTICLES_DATA_FILE, NPY)
        maxNumCoords = self.numberOfCoords.get()
        codePatch = f""" 
import emfile 
import numpy as np

_, data = emfile.read('{particlesEmFile}')
# Use numpy.squeeze() to remove dimensions of size 1 
data = np.squeeze(data)
# Save the first N particles, knowing that they are sorted descending by score
if {maxNumCoords} > -1 and data.shape[0] > {maxNumCoords}:
    data = data[0:{maxNumCoords}, :]
np.save('{outParticlesNpyFile}', data)
        """
        readEmParticlePythonFile = join(self._getCurrentTomoDir(tsId), 'readEmParticles.py')
        with open(readEmParticlePythonFile, "w") as pyFile:
            pyFile.write(codePatch)
        Plugin.runGapStop(self, PYTHON, readEmParticlePythonFile, isCryoCatExec=True)
        return outParticlesNpyFile

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
