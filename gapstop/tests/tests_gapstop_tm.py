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
from os.path import exists

from xmipp3.protocols import XmippProtCropResizeVolumes, XmippProtCreateMask3D
from gapstop.protocols import ProtGapStopTemplateMatching, ProtGapStopExtractCoords
from pwem.protocols import ProtImportVolumes
from pyworkflow.tests import setupTestProject, DataSet
from pyworkflow.utils import magentaStr, cyanStr
from tomo.protocols import ProtImportTs, ProtImportTsCTF, ProtImportTomograms
from tomo.protocols.protocol_import_ctf import ImportChoice
from tomo.protocols.protocol_import_tomograms import OUTPUT_NAME
from tomo.tests import RE4_STA_TUTO, DataSetRe4STATuto
from tomo.tests.test_base_centralized_layer import TestBaseCentralizedLayer


class TestGapStopTM(TestBaseCentralizedLayer):
    unbinnedSRate = DataSetRe4STATuto.unbinnedPixSize.value
    binFactor4 = 4
    binFactor8 = 8
    nTomos = 2
    expectedTomoDims = [480, 464, 140]

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet(RE4_STA_TUTO)
        cls._runPreviousProtocols()

    @classmethod
    def _runPreviousProtocols(cls):
        print(cyanStr('--------------------------------- RUNNING PREVIOUS PROTOCOLS ---------------------------------'))
        cls.importedTs = cls._runImportTs()
        cls.importedCtfs = cls._runImportCtf()
        cls.tomoNoFidBin8 = cls._runImportTomograms()
        importedRefBin4 = cls._runImportReference()
        cls.refBin8 = cls._runCropResizeVolBin8(importedRefBin4)
        cls.maskBin8 = cls._runCreateMask3D()
        print(
            cyanStr('\n-------------------------------- PREVIOUS PROTOCOLS FINISHED ---------------------------------'))

    @classmethod
    def _runImportTs(cls):
        print(magentaStr("\n==> Importing the tilt series:"))
        protImportTs = cls.newProtocol(ProtImportTs,
                                       filesPath=cls.ds.getFile(DataSetRe4STATuto.tsPath.value),
                                       filesPattern=DataSetRe4STATuto.tsPattern.value,
                                       exclusionWords=DataSetRe4STATuto.exclusionWordsTs03ts54.value,
                                       anglesFrom=2,  # From tlt file
                                       voltage=DataSetRe4STATuto.voltage.value,
                                       magnification=DataSetRe4STATuto.magnification.value,
                                       sphericalAberration=DataSetRe4STATuto.sphericalAb.value,
                                       amplitudeContrast=DataSetRe4STATuto.amplitudeContrast.value,
                                       samplingRate=cls.unbinnedSRate,
                                       doseInitial=DataSetRe4STATuto.initialDose.value,
                                       dosePerFrame=DataSetRe4STATuto.dosePerTiltImg.value,
                                       tiltAxisAngle=DataSetRe4STATuto.tiltAxisAngle.value)

        cls.launchProtocol(protImportTs)
        tsImported = getattr(protImportTs, 'outputTiltSeries', None)
        return tsImported

    @classmethod
    def _runImportCtf(cls):
        print(magentaStr("\n==> Importing the CTFs:"))
        protImportCtf = cls.newProtocol(ProtImportTsCTF,
                                        filesPath=cls.ds.getFile(DataSetRe4STATuto.tsPath.value),
                                        filesPattern=DataSetRe4STATuto.ctfPattern.value,
                                        importFrom=ImportChoice.CTFFIND.value,
                                        inputSetOfTiltSeries=cls.importedTs)
        cls.launchProtocol(protImportCtf)
        outputMask = getattr(protImportCtf, protImportCtf._possibleOutputs.CTFs.name, None)
        return outputMask

    @classmethod
    def _runImportTomograms(cls):
        print(magentaStr("\n==> Importing the tomograms:"))
        protImportTomos = cls.newProtocol(ProtImportTomograms,
                                          filesPath=cls.ds.getFile(DataSetRe4STATuto.tomogramsNoFidPath.value),
                                          filesPattern='*.mrc',
                                          samplingRate=cls.unbinnedSRate * cls.binFactor8)  # Bin 8
        cls.launchProtocol(protImportTomos)
        outTomos = getattr(protImportTomos, OUTPUT_NAME, None)
        return outTomos

    @classmethod
    def _runImportReference(cls):
        print(magentaStr("\n==> Importing the reference volume:"))
        protImportRef = cls.newProtocol(ProtImportVolumes,
                                        filesPath=cls.ds.getFile(DataSetRe4STATuto.initModelRelion.name),
                                        samplingRate=cls.unbinnedSRate * cls.binFactor4)
        cls.launchProtocol(protImportRef)
        return getattr(protImportRef, ProtImportVolumes._possibleOutputs.outputVolume.name, None)

    @classmethod
    def _runCropResizeVolBin8(cls, inVol):
        print(magentaStr("\n==> Resizing the reference to bin 8:"))
        protCropResize = cls.newProtocol(XmippProtCropResizeVolumes,
                                         inputVolumes=inVol,
                                         doResize=True,
                                         resizeOption=0,  # RESIZE_SAMPLINGRATE,
                                         resizeSamplingRate=cls.unbinnedSRate * cls.binFactor8)
        cls.launchProtocol(protCropResize)
        outputMask = getattr(protCropResize, 'outputVol', None)
        return outputMask
    
    @classmethod
    def _runCreateMask3D(cls):
        print(magentaStr("\n==> Creating the reference mask:"))
        protCreateMask = cls.newProtocol(XmippProtCreateMask3D,
                                         source=1,  # Geometry
                                         samplingRate=cls.unbinnedSRate * cls.binFactor8,
                                         size=48,
                                         geo=3,  # Cylinder
                                         radius=18,
                                         height=18,
                                         sigmaConvolution=3)
        cls.launchProtocol(protCreateMask)
        return getattr(protCreateMask, 'outputMask', None)

    def testGapStopTM(self):
        print(magentaStr("\n==> Running the GapStop_TM:"))
        sRateBin8 = self.unbinnedSRate * self.binFactor8
        protGapStopTM = self.newProtocol(ProtGapStopTemplateMatching,
                                         inTomos=self.tomoNoFidBin8,
                                         inCtfSet=self.importedCtfs,
                                         inTsSet=self.importedTs,
                                         reference=self.refBin8,
                                         mask=self.maskBin8,
                                         doInvertRefContrast=True,
                                         nTiles=8,
                                         coneSampling=15,
                                         rotSymDeg=6)
        self.launchProtocol(protGapStopTM)
        scoreTomos = getattr(protGapStopTM, protGapStopTM._possibleOutputs.scoreTomogrmas.name, None)
        # Check the results of the gapStop_TM
        self.checkTomograms(inTomoSet=scoreTomos,
                            expectedSetSize=self.nTomos,
                            expectedSRate=sRateBin8,
                            expectedDimensions=self.expectedTomoDims)
        # GapStopScoreTomogram specific attributes
        for tomo in scoreTomos:
            self.assertTrue(exists(tomo.getTomoFile()))
            self.assertTrue(exists(tomo.getAnglesMap()))
            self.assertTrue(exists(tomo.getAngleList()))
            self.assertGreater(tomo.getTomoNum(), 0)
            self.assertEqual(tomo.getSymmetry(), 'C6')

        print(magentaStr("\n==> GapStop_TM == > extracting the coordinates"))
        particleDiameter = 10
        protGapStopExtract = self.newProtocol(ProtGapStopExtractCoords,
                                              inScoreTomos=scoreTomos,
                                              scoresThreshold=0.09,
                                              percentile=99.9,
                                              partDiameter=particleDiameter,
                                              numberOfCoords=-1)
        self.launchProtocol(protGapStopExtract)
        coords = getattr(protGapStopExtract, protGapStopExtract._possibleOutputs.coordinates.name, None)
        # Check the results of gapStop's extraction
        self.checkCoordinates(outCoords=coords,
                              expectedBoxSize=particleDiameter,
                              expectedSRate=sRateBin8,
                              orientedParticles=True,
                              orientedTolPercent=0.01)  # 1%
        self.assertTrue(coords.getSize() > 2000)
