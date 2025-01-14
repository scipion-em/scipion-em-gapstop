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
from typing import Union

from gapstop import Plugin
from pwem.objects import VolumeMask
from pwem.protocols import EMProtocol
from pyworkflow import BETA
from pyworkflow.object import Pointer
from pyworkflow.protocol import STEPS_PARALLEL, PointerParam, FloatParam, StringParam, IntParam, GPU_LIST
from pyworkflow.utils import Message, makePath
from scipion.constants import PYTHON
from tomo.objects import SetOfCoordinates3D, SetOfTomograms

logger = logging.getLogger(__name__)
IN_TOMOS = 'inTomos'
REF_VOL = 'refVol'
IN_MASK = 'inMask'


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

    # --------------------------- DEFINE param functions ----------------------
    def _defineParams(self, form):
        form.addSection(label=Message.LABEL_INPUT)
        form.addParam(IN_TOMOS, PointerParam,
                      pointerClass='SetOfTomograms',
                      important=True,
                      label='Tomograms')
        form.addParam(REF_VOL, PointerParam,
                      pointerClass='Volume',
                      important=True,
                      label="Reference volume")
        form.addParam(IN_MASK, PointerParam,
                      pointerClass=VolumeMask,
                      important=True,
                      label='Reference mask')
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
        self._insertFunctionStep(self.prepareAnglesStep,
                                 prerequisites=[],
                                 needsGPU=False)
        # for tomo in self._getInTomoSet().iterItems():
        #     tsId = tomo.getTsId()
        #     cInputId = self._insertFunctionStep(self.convertInputStep, tsId,
        #                                         prerequisites=[],
        #                                         needsGPU=False)

    # -------------------------- STEPS functions ------------------------------
    def _initialize(self):
        self.sRate = self._getInTomoSet().getSamplingRate()

    def prepareAnglesStep(self):
        angleListFile = self._getCryoCatAngleFile()
        codePatch = f""" 
        from cryocat import geom 
        import numpy as np 
        angles = geom.generate_angles({self.coneAngle.get()}, {self.coneSampling.get()}, symmetry={self.rotSymDeg.get()}) 
        np.savetxt({angleListFile}, angles, fmt='%.2f', delimiter=',') """

        genAnglesPythonFile = self._getExtraPath('prepAngles.py')
        with open(genAnglesPythonFile, "w") as pyFile:
            pyFile.write(codePatch)
        Plugin.runGapStop(self, PYTHON, genAnglesPythonFile, isCryoCatExec=True)

    def convertInputStep(self, tsId: str):
        tsDir = self._getCurrentTomoDir(tsId)
        makePath(tsDir)

    # --------------------------- UTILS functions -----------------------------
    def _getCryoCatAngleFile(self):
        return self._getExtraPath(f'angles_{self.coneSampling.get():.0f}_c{self.rotSymDeg.get()}.txt')

    def _getInTomoSet(self, returnPointer: bool = False) -> Union[SetOfTomograms, Pointer]:
        inTsPointer = getattr(self, IN_TOMOS)
        return inTsPointer if returnPointer else inTsPointer.get()

    def _getCurrentTomoDir(self, tsId: str) -> str:
        return self._getExtraPath(tsId)