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
from typing import Union, List, Tuple

from cistem.protocols import CistemProtTsCtffind
from gapstop.objects import SetOfGapStopScoreTomograms
from imod.constants import OUTPUT_TILTSERIES_NAME
from imod.protocols import ProtImodExcludeViews
from pwem.objects import VolumeMask, Volume
from tomo.objects import SetOfCTFTomoSeries, SetOfTiltSeries, SetOfCoordinates3D, TiltSeries, CTFTomoSeries, \
    SetOfTomograms
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

TS_03 = 'TS_03'
TS_54 = 'TS_54'


class TestGapStopTM(TestBaseCentralizedLayer):
    unbinnedSRate = DataSetRe4STATuto.unbinnedPixSize.value
    binFactor4 = 4
    binFactor8 = 8
    nTomos = 2
    expectedTomoDims = [480, 464, 140]
    excludedViewsDict = {
        TS_03: [0, 1, 2, 38, 39],
        TS_54: [0, 1, 38, 39, 40]
    }
    ctfExcludedViewsDict = {
        TS_03: [0, 1, 38, 39],
        TS_54: [0, 39, 40]
    }
    UNMODIFIED = 'unmodified'
    EXC_VIEWS = 'exc. views'
    RE_STACKED = 're-stacked'
    particleDiameter = 10

    @classmethod
    def setUpClass(cls):
        setupTestProject(cls)
        cls.ds = DataSet.getDataSet(RE4_STA_TUTO)
        cls.sRateBin8 = cls.unbinnedSRate * cls.binFactor8

    @classmethod
    def _runPreviousProtocols(cls, eVCtf: bool = False, eVTs: bool = False) \
            -> Tuple[SetOfTiltSeries, SetOfCTFTomoSeries]:
        print(cyanStr('--------------------------------- RUNNING PREVIOUS PROTOCOLS ---------------------------------'))
        importedTs = cls._runImportTs()
        if eVTs:
            cls._excludeSetViews(importedTs)
        importedCtfs = cls._runImportCtf(importedTs)
        if eVCtf:
            cls._excludeSetViews(importedCtfs, excludedViewsDict=cls.ctfExcludedViewsDict)
        cls.tomoNoFidBin8 = cls._runImportTomograms()
        importedRefBin4 = cls._runImportReference()
        cls.refBin8 = cls._runCropResizeVolBin8(importedRefBin4)
        cls.maskBin8 = cls._runCreateMask3D()
        print(
            cyanStr('\n-------------------------------- PREVIOUS PROTOCOLS FINISHED ---------------------------------'))
        return importedTs, importedCtfs

    @classmethod
    def _excludeSetViews(cls,
                         inSet: Union[SetOfTiltSeries, SetOfCTFTomoSeries],
                         excludedViewsDict: Union[dict, None] = None) -> None:
        if not excludedViewsDict:
            excludedViewsDict = cls.excludedViewsDict
        objList = [obj.clone(ignoreAttrs=[]) for obj in inSet]
        for obj in objList:
            cls._excIntermediateSetViews(inSet, obj, excludedViewsDict[obj.getTsId()])

    @staticmethod
    def _excIntermediateSetViews(inSet: Union[SetOfTiltSeries, SetOfCTFTomoSeries],
                                 obj: Union[TiltSeries, CTFTomoSeries],
                                 excludedViewsList: List[int]) -> None:
        tiList = [ti.clone() for ti in obj]
        for i, ti in enumerate(tiList):
            if i in excludedViewsList:
                ti._objEnabled = False
                obj.update(ti)
        obj.write()
        inSet.update(obj)
        inSet.write()
        inSet.close()

    @classmethod
    def _runImportTs(cls) -> SetOfTiltSeries:
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
    def _runImportCtf(cls, importedTs: SetOfTiltSeries) -> SetOfCTFTomoSeries:
        print(magentaStr("\n==> Importing the CTFs:"))
        protImportCtf = cls.newProtocol(ProtImportTsCTF,
                                        filesPath=cls.ds.getFile(DataSetRe4STATuto.tsPath.value),
                                        filesPattern=DataSetRe4STATuto.ctfPattern.value,
                                        importFrom=ImportChoice.CTFFIND.value,
                                        inputSetOfTiltSeries=importedTs)
        cls.launchProtocol(protImportCtf)
        outputCtfs = getattr(protImportCtf, protImportCtf._possibleOutputs.CTFs.name, None)
        return outputCtfs

    @classmethod
    def _runImportTomograms(cls) -> SetOfTomograms:
        print(magentaStr("\n==> Importing the tomograms:"))
        protImportTomos = cls.newProtocol(ProtImportTomograms,
                                          filesPath=cls.ds.getFile(DataSetRe4STATuto.tomogramsNoFidPath.value),
                                          filesPattern='*.mrc',
                                          samplingRate=cls.unbinnedSRate * cls.binFactor8)  # Bin 8
        cls.launchProtocol(protImportTomos)
        outTomos = getattr(protImportTomos, OUTPUT_NAME, None)
        return outTomos

    @classmethod
    def _runImportReference(cls) -> Volume:
        print(magentaStr("\n==> Importing the reference volume:"))
        protImportRef = cls.newProtocol(ProtImportVolumes,
                                        filesPath=cls.ds.getFile(DataSetRe4STATuto.initModelRelion.name),
                                        samplingRate=cls.unbinnedSRate * cls.binFactor4)
        cls.launchProtocol(protImportRef)
        return getattr(protImportRef, ProtImportVolumes._possibleOutputs.outputVolume.name, None)

    @classmethod
    def _runCropResizeVolBin8(cls, inVol: Volume) -> VolumeMask:
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
    def _runCreateMask3D(cls) -> VolumeMask:
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

    @classmethod
    def _runExcludeViewsProt(cls,
                             inTsSet: SetOfTiltSeries,
                             objLabel: str = None) -> SetOfTiltSeries:
        print(magentaStr("\n==> Running the TS exclusion of views:"))
        protExcViews = cls.newProtocol(ProtImodExcludeViews, inputSetOfTiltSeries=inTsSet)
        if objLabel:
            protExcViews.setObjLabel(objLabel)
        cls.launchProtocol(protExcViews)
        outTsSet = getattr(protExcViews, OUTPUT_TILTSERIES_NAME, None)
        return outTsSet

    @classmethod
    def _runCistemEstimateCtf(cls, inTsSet: SetOfTiltSeries) -> SetOfCTFTomoSeries:
        print(magentaStr("\n==> Estimating the CTF with Cistem:"))
        protEstimateCtf = cls.newProtocol(CistemProtTsCtffind,
                                          inputTiltSeries=inTsSet,
                                          lowRes=50,
                                          highRes=5,
                                          minDefocus=5000,
                                          maxDefocus=50000)
        cls.launchProtocol(protEstimateCtf)
        return getattr(protEstimateCtf, CistemProtTsCtffind._possibleOutputs.CTFs.name, None)

    @classmethod
    def _genReStackedCtf(cls) -> SetOfCTFTomoSeries:
        tsSet = cls._runImportTs()
        # Exclude some views from the TS at metadata level
        cls._excludeSetViews(tsSet, excludedViewsDict=cls.ctfExcludedViewsDict)
        # Re-stack that TS
        reStackedTsSet = cls._runExcludeViewsProt(tsSet)
        # Estimate the CTF using the re-stacked TS
        return cls._runCistemEstimateCtf(reStackedTsSet)

    def _runGapStopTM(self,
                      inCtfSet: SetOfCTFTomoSeries,
                      inTsSet: SetOfTiltSeries,
                      ctfSetMsg: str,
                      tsSetMsg: str) \
            -> Union[SetOfGapStopScoreTomograms, None]:
        print(magentaStr(f"\n==> Running the GapStop_TM:"
                         f"\n\t- CTFs: {ctfSetMsg}"
                         f"\n\t- Tilt-series = {tsSetMsg}"))
        protGapStopTM = self.newProtocol(ProtGapStopTemplateMatching,
                                         inTomos=self.tomoNoFidBin8,
                                         inCtfSet=inCtfSet,
                                         inTsSet=inTsSet,
                                         reference=self.refBin8,
                                         mask=self.maskBin8,
                                         doInvertRefContrast=True,
                                         nTiles=8,
                                         coneSampling=15,
                                         rotSymDeg=6)
        objLabel = f'ts {tsSetMsg}, ctf {ctfSetMsg}'
        protGapStopTM.setObjLabel(objLabel)
        self.launchProtocol(protGapStopTM)
        return getattr(protGapStopTM, protGapStopTM._possibleOutputs.scoreTomogrmas.name, None)

    def _checkScoredTomos(self, scoreTomos: SetOfGapStopScoreTomograms) -> None:
        # Check the results of the gapStop_TM
        self.checkTomograms(inTomoSet=scoreTomos,
                            expectedSetSize=self.nTomos,
                            expectedSRate=self.sRateBin8,
                            expectedDimensions=self.expectedTomoDims)
        # GapStopScoreTomogram specific attributes
        for tomo in scoreTomos:
            self.assertTrue(exists(tomo.getTomoFile()))
            self.assertTrue(exists(tomo.getAnglesMap()))
            self.assertTrue(exists(tomo.getAngleList()))
            self.assertGreater(tomo.getTomoNum(), 0)
            self.assertEqual(tomo.getSymmetry(), 'C6')

    def _runGapStopExtractCoords(self, scoreTomos: SetOfGapStopScoreTomograms) \
            -> Union[SetOfCoordinates3D, None]:
        print(magentaStr("\n==> GapStop_TM == > extracting the coordinates"))
        protGapStopExtract = self.newProtocol(ProtGapStopExtractCoords,
                                              inScoreTomos=scoreTomos,
                                              scoresThreshold=0.09,
                                              percentile=99.9,
                                              partDiameter=self.particleDiameter,
                                              numberOfCoords=-1)
        self.launchProtocol(protGapStopExtract)
        return getattr(protGapStopExtract, protGapStopExtract._possibleOutputs.coordinates.name, None)

    def _checkExtractedCoords(self, coords: SetOfCoordinates3D) -> None:
        self.checkCoordinates(outCoords=coords,
                              expectedBoxSize=self.particleDiameter,
                              expectedSRate=self.sRateBin8,
                              orientedParticles=True,
                              orientedTolPercent=0.01)  # 1%
        self.assertTrue(coords.getSize() > 2000)

    def _runTestGapStopTM(self,
                      inCtfSet: SetOfCTFTomoSeries,
                      inTsSet: SetOfTiltSeries,
                      ctfSetMsg: str,
                      tsSetMsg: str) -> None:
        # Run the template matching
        scoreTomos = self._runGapStopTM(inCtfSet=inCtfSet,
                                        inTsSet=inTsSet,
                                        ctfSetMsg=ctfSetMsg,
                                        tsSetMsg=tsSetMsg,)
        # Check the scored tomograms
        self._checkScoredTomos(scoreTomos)
        # Run the coordinate extraction
        coords = self._runGapStopExtractCoords(scoreTomos)
        # Check the results of gapStop's extraction
        self._checkExtractedCoords(coords)

    def testGapStop(self):
        importedTs, importedCtfs = self._runPreviousProtocols()
        self._runTestGapStopTM(inCtfSet=importedCtfs,
                               inTsSet=importedTs,
                               ctfSetMsg=self.UNMODIFIED,
                               tsSetMsg=self.UNMODIFIED)

    def testGapStop_EV_Ctf(self):
        importedTs, importedCtfs = self._runPreviousProtocols(eVCtf=True)
        self._runTestGapStopTM(inCtfSet=importedCtfs,
                               inTsSet=importedTs,
                               ctfSetMsg=self.EXC_VIEWS,
                               tsSetMsg=self.UNMODIFIED)

    def testGapStop_EV_Ts(self):
        importedTs, importedCtfs = self._runPreviousProtocols(eVTs=True)
        self._runTestGapStopTM(inCtfSet=importedCtfs,
                               inTsSet=importedTs,
                               ctfSetMsg=self.UNMODIFIED,
                               tsSetMsg=self.EXC_VIEWS)

    def testGapStop_EV_Restacked_Ctf(self):
        importedTs, _ = self._runPreviousProtocols()
        ctfSetReStacked = self._genReStackedCtf()  # Gen a CTF estimated on a re-stacked TS
        self._runTestGapStopTM(inCtfSet=ctfSetReStacked,
                               inTsSet=importedTs,
                               ctfSetMsg=self.EXC_VIEWS,
                               tsSetMsg=self.UNMODIFIED)

    def testGapStop_EV_Restacked_Ts(self):
        importedTs, importedCtfs = self._runPreviousProtocols(eVTs=True)
        tsSetReStacked = self._runExcludeViewsProt(importedTs)  # Re-stack the TS
        self._runTestGapStopTM(inCtfSet=importedCtfs,
                               inTsSet=tsSetReStacked,
                               ctfSetMsg=self.UNMODIFIED,
                               tsSetMsg=self.EXC_VIEWS)