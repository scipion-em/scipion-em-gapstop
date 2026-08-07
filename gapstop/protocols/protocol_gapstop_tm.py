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
import traceback
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
from pwem.convert.headers import setMRCSamplingRate
from pwem.emlib.image import ImageHandler
from pwem.objects import VolumeMask, Volume
from pyworkflow import BETA
from pyworkflow.object import Set, String
from pyworkflow.protocol import PointerParam, FloatParam, StringParam, IntParam, GPU_LIST, BooleanParam, \
    LEVEL_ADVANCED
from pyworkflow.utils import Message, makePath, getExt, createLink, cyanStr, redStr
import itertools
from scipion.constants import PYTHON
from tomo.objects import SetOfTiltSeries, CTFTomo
from tomo.utils import getObjFromRelation, getCommonTsAndCtfElements

logger = logging.getLogger(__name__)


class GapStopTMOutputs(Enum):
    scoreTomogrmas = SetOfGapStopScoreTomograms


class ProtGapStopTemplateMatching(ProtGapStopBase):
    """
    Performs GPU-accelerated template matching on cryo-electron tomography datasets using
    GAPSTOPTM. The protocol detects candidate molecular complexes inside tomograms by
    comparing a reference structure against volumetric data over a wide range of orientations
    and spatial positions.

    AI Generated:

    GapStop Template Matching (ProtGapStopTemplateMatching) - User Manual
        Overview

        The GapStop Template Matching protocol is designed for large-scale template matching
        in cryo-electron tomography workflows. Its purpose is to identify regions inside
        tomograms that resemble a known structural template, allowing researchers to locate
        macromolecular assemblies directly within crowded cellular environments. The protocol
        is optimized for modern GPU clusters and distributed high-performance computing
        systems, making it suitable for demanding in situ structural biology projects.

        In practical biological applications, template matching is commonly used to detect
        ribosomes, membrane-bound complexes, viral particles, cytoskeletal assemblies, or
        other macromolecules whose approximate structure is already known. Instead of
        reconstructing particles individually before localization, the protocol searches the
        tomogram directly and generates volumetric score maps that estimate how well the
        template matches every possible position and orientation.

        The protocol is particularly valuable in workflows involving large tomograms or large
        collections of tomographic reconstructions, where exhaustive searches would otherwise
        become computationally prohibitive. By combining GPU acceleration with parallel
        processing strategies, the protocol enables practical execution of searches that would
        traditionally require very long runtimes.

        Inputs and Experimental Context

        The protocol requires a set of tomograms together with a reference volume representing
        the expected molecular structure. In many cryo-ET studies, this reference originates
        from subtomogram averaging, single-particle cryo-EM, or atomic modeling workflows.
        The closer the reference resembles the biological target, the more reliable the
        resulting detections are expected to be.

        Optionally, tilt-series and CTF information can also be provided. These inputs improve
        the physical realism of the matching process because they allow the protocol to model
        missing wedge effects and acquisition-dependent imaging properties. This becomes
        particularly important in high-resolution or low-contrast datasets where imaging
        artifacts strongly influence detectability.

        Biological users should ensure that tomograms, references, and masks are all
        compatible in sampling rate and overall scale. Mismatches in voxel size or spatial
        dimensions may lead to poor localization accuracy or biologically misleading results.

        Reference Contrast and Biological Interpretation

        A critical aspect of template matching is the contrast relationship between the
        template and the tomograms. Cryo-electron tomography datasets frequently display
        biological densities as dark features, whereas reference maps from other workflows may
        appear inverted. The protocol therefore allows contrast inversion of the template when
        needed.

        From a biological perspective, proper contrast matching is essential because template
        matching relies on correlation between structural features. Incorrect contrast
        orientation may suppress true detections and artificially enhance noise. When using
        references imported from external reconstruction software, users should visually
        verify that densities appear with the same polarity as in the tomograms before
        starting large-scale searches.

        Masking and Search Focus

        The reference mask is one of the most biologically important components of the
        workflow. The mask defines which regions of the template contribute to the matching
        calculation and therefore determines the structural features emphasized during the
        search.

        Compact masks focused on stable structural cores usually provide the most robust
        detections. For example, when searching for flexible membrane complexes, excluding
        highly mobile peripheral regions often improves localization accuracy substantially.
        In contrast, excessively large masks may include solvent or unrelated densities that
        dilute the specificity of the correlation scores.

        The protocol also uses the mask to restrict the interpretation of score maps,
        ensuring that the reported signals correspond primarily to biologically relevant
        structural regions. Thoughtful mask design is therefore essential for reliable
        downstream analysis.

        Angular Sampling Strategies

        Template matching requires systematic exploration of molecular orientations. The
        protocol provides both standard and fully customizable angular sampling schemes. In
        routine applications, users typically define a cone angle and angular sampling step,
        which together determine how densely orientation space is explored.

        Wider angular coverage increases the likelihood of detecting molecules in arbitrary
        orientations but also increases computational cost. Finer angular sampling improves
        orientation precision but may substantially increase runtime. Biological users should
        therefore balance sensitivity against computational feasibility according to the
        expected structural variability of the target.

        Advanced users may define explicit angular ranges for individual Euler angles. This
        becomes particularly useful when prior biological knowledge limits the expected
        orientations of a complex. For example, membrane-associated proteins may preferentially
        adopt constrained orientations relative to membranes, allowing users to reduce the
        search space and accelerate processing.

        Symmetry Considerations

        Rotational symmetry can significantly improve both efficiency and robustness during
        template matching. When the target complex possesses known cyclic symmetry, the
        protocol can incorporate this information directly into the search process.

        Correct symmetry specification reduces redundant orientation sampling and strengthens
        biologically meaningful correlations. However, incorrect symmetry assumptions may
        artificially bias detections or hide asymmetric structural features. Users should only
        apply symmetry constraints when they are well supported by structural evidence.

        Template Filtering and Resolution Control

        The protocol allows low-pass and high-pass filtering of the template prior to
        matching. These filters control which spatial frequencies contribute to the search
        and therefore influence both sensitivity and specificity.

        In most biological applications, moderate low-pass filtering improves robustness by
        emphasizing large-scale structural features while suppressing high-frequency noise.
        High-pass filtering is generally less critical but may help reduce large-scale
        background variations in certain datasets.

        The optimal filtering strategy depends strongly on tomogram quality, expected target
        size, and biological heterogeneity. Flexible or partially disordered complexes often
        benefit from lower-resolution searches focused on stable global architecture rather
        than fine structural details.

        Tomogram Tiling and Large-Scale Processing

        Large tomograms can exceed GPU memory limits during template matching. To address this,
        the protocol supports tomogram decomposition into smaller computational tiles. This
        strategy enables efficient processing of very large cellular volumes while preserving
        scalability across multiple GPUs and compute nodes.

        Increasing the number of tiles may reduce memory pressure and improve execution
        stability, although excessively fine decomposition can increase runtime overhead.
        Biological users working with thick cellular tomograms or very large fields of view
        commonly rely on tiling strategies to maintain practical performance.

        Outputs and Their Interpretation

        The primary outputs are score tomograms and orientation maps. The score maps indicate
        how strongly the reference matches each spatial location inside the tomogram, while
        the orientation maps record the angular assignment associated with each detection.

        High-scoring regions represent candidate molecular localizations that can be further
        analyzed or converted into coordinate sets for downstream subtomogram extraction and
        averaging workflows. However, biological interpretation requires caution because high
        scores do not automatically guarantee true positives. Crowded environments, repetitive
        structures, and imaging artifacts can produce misleading correlations.

        For this reason, template matching results are typically followed by additional
        validation steps such as coordinate clustering, subtomogram averaging, classification,
        or manual inspection. Biological confidence emerges from the consistency of evidence
        across multiple stages of analysis rather than from correlation values alone.

        Practical Recommendations

        In routine cryo-ET workflows, users should begin with conservative angular sampling
        and moderate filtering parameters to establish baseline detections. If targets are
        difficult to identify, improving the mask or adjusting the angular search density
        often provides larger gains than aggressively increasing computational complexity.

        When working with noisy cellular tomograms, lower-resolution templates frequently
        outperform highly detailed references because they emphasize robust structural
        signatures. Similarly, introducing biologically informed orientation constraints may
        greatly improve both speed and specificity.

        Users processing large datasets on GPU clusters should carefully monitor memory usage
        and consider increasing tomogram tiling when failures occur. Binning tomograms before
        matching is also a common strategy for exploratory searches or for very large cellular
        reconstructions.

        Final Perspective

        Template matching is one of the central techniques in in situ structural biology
        because it bridges the gap between molecular structure determination and cellular
        context. The GapStop Template Matching protocol provides a scalable and biologically
        oriented framework for detecting macromolecular assemblies directly inside tomograms
        while leveraging modern GPU computing resources.

        Reliable results depend not only on computational performance but also on thoughtful
        biological decisions regarding template selection, masking strategy, angular sampling,
        filtering, and interpretation of score maps. When used carefully, the protocol enables
        powerful exploration of molecular organization within native cellular environments.
    """

    _label = 'template matching'
    _devStatus = BETA
    _possibleOutputs = GapStopTMOutputs
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
        # Custom angular sampling parameters
        form.addParam('useCustomAngSampling', BooleanParam,
                      default=False,
                      label='Set custom angular sampling',
                      help='If set to Yes, you can define custom angular ranges and steps for each Euler angle '
                           '(alpha, beta, gamma) in ZXZ convention. This overrides the cone angle and cone sampling '
                           'parameters above.')
        groupCustomAng = form.addGroup('Custom angular ranges',
                                       condition='useCustomAngSampling')
        groupCustomAng.addParam('alphaStart', FloatParam,
                                default=0,
                                label='Alpha start (deg.)',
                                help='Start angle for alpha (Z rotation). Range: [0, 360)')
        groupCustomAng.addParam('alphaEnd', FloatParam,
                                default=360,
                                label='Alpha end (deg.)',
                                help='End angle for alpha (Z rotation). Range: (0, 360]')
        groupCustomAng.addParam('alphaStep', FloatParam,
                                default=10,
                                label='Alpha step (deg.)',
                                help='Step size for alpha angle sampling.')
        groupCustomAng.addParam('betaStart', FloatParam,
                                default=40,
                                label='Beta start (deg.)',
                                help='Start angle for beta (X tilt). Range: [0, 180]. '
                                     'For example, 40-140 leaves ±50° missing caps.')
        groupCustomAng.addParam('betaEnd', FloatParam,
                                default=140,
                                label='Beta end (deg.)',
                                help='End angle for beta (X tilt). Range: [0, 180]. '
                                     'For example, 40-140 leaves ±50° missing caps.')
        groupCustomAng.addParam('betaStep', FloatParam,
                                default=10,
                                label='Beta step (deg.)',
                                help='Step size for beta angle sampling.')
        groupCustomAng.addParam('gammaStart', FloatParam,
                                default=0,
                                label='Gamma start (deg.)',
                                help='Start angle for gamma (Z rotation). Range: [0, 360)')
        groupCustomAng.addParam('gammaEnd', FloatParam,
                                default=360,
                                label='Gamma end (deg.)',
                                help='End angle for gamma (Z rotation). Range: (0, 360]')
        groupCustomAng.addParam('gammaStep', FloatParam,
                                default=10,
                                label='Gamma step (deg.)',
                                help='Step size for gamma angle sampling.')
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

    # --------------------------- INSERT steps functions ----------------------
    def _insertAllSteps(self):
        self._initialize()
        closeSetStepDeps = []
        cRId = self._insertFunctionStep(self.convertReferenceStep,
                                        prerequisites=[],
                                        needsGPU=False)
        # Choose the appropriate angle preparation step based on user choice
        if self.useCustomAngSampling.get():
            pAngId = self._insertFunctionStep(self.prepareCustomAngStep,
                                              prerequisites=cRId,
                                              needsGPU=False)
        else:
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
        unionTsIds = tomosTsIds | tsIds | ctfTsIds
        nonMatchingTsIds = presentTsIds - unionTsIds

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
        logger.info(cyanStr(f"Converting the reference in the required format...'"))
        try:
            # Convert or link the reference
            ref = self._getFormAttrib(REF_VOL)
            if self.doInvertRefContrast.get():
                self._invertReference(ref)
            else:
                self._convertOrLinkVolume(ref, self.refName)
            # Convert or link the mask
            mask = self._getFormAttrib(IN_MASK)
            self._convertOrLinkVolume(mask, self.maskName)
        except Exception as e:
            raise Exception(f'Reference conversion failed with the exception -> {e}')

    def prepareAnglesStep(self):
        try:
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
        except Exception as e:
            raise Exception(f'Angles file generation with the exception -> {e}')

    def prepareCustomAngStep(self):
        """Generate custom angular sampling for template matching (ZXZ convention, degrees).
        User defines custom ranges and steps for each Euler angle (alpha, beta, gamma)."""
        try:
            logger.info(cyanStr('Generating the file with custom Euler angles specifying the rotations...'))
            angleListFile = self._getCustomAngleFile()

            # Get user-defined ranges (end values are exclusive in arange, so we add step to include end)
            alphaStart = self.alphaStart.get()
            alphaEnd = self.alphaEnd.get()
            alphaStep = self.alphaStep.get()
            betaStart = self.betaStart.get()
            betaEnd = self.betaEnd.get()
            betaStep = self.betaStep.get()
            gammaStart = self.gammaStart.get()
            gammaEnd = self.gammaEnd.get()
            gammaStep = self.gammaStep.get()

            # Generate angle arrays in degrees
            alpha_deg = np.arange(alphaStart, alphaEnd + alphaStep, alphaStep)  # +step to include end
            beta_deg = np.arange(betaStart, betaEnd + betaStep, betaStep)  # +step to include end
            gamma_deg = np.arange(gammaStart, gammaEnd + gammaStep, gammaStep)  # +step to include end

            # Generate all combinations (keep in degrees)
            angles = []
            for a_deg, b_deg, g_deg in itertools.product(alpha_deg, beta_deg, gamma_deg):
                angles.append([a_deg, b_deg, g_deg])

            angles = np.array(angles)
            np.savetxt(angleListFile, angles, fmt='%.2f', delimiter=',')
            logger.info(cyanStr(f'Generated {len(angles)} angular triplets in {angleListFile}'))
        except Exception as e:
            raise Exception(f'Custom angles file generation failed with the exception -> {e}')

    def convertInputStep(self, tsId: str):
        try:
            tomo = self.tomoDict[tsId]
            ts = self.tsDict[tsId]
            ctf = self.ctfDict[tsId]
            presentAcqOrders = getCommonTsAndCtfElements(ts, ctf)
            if len(presentAcqOrders) == 0:
                raise Exception(f'tsId = {tsId} -> No common acquisition orders found between the '
                                f'tilt-series and the CTF.')

            logger.info(cyanStr(f"tsId = {tsId} -> present acquisition orders in both "
                                f"the tilt-series and the CTF are {presentAcqOrders}.'"))
            
            acq = ts.getAcquisition()
            tomoObjId = tomo.getObjId()
            tsDir = self._getCurrentTomoDir(tsId)
            makePath(tsDir)

            # Convert or link the current tomogram
            logger.info(cyanStr(f'tsId = {tsId}: converting or linking the tomogram...'))
            inTomoName = self._getWorkingTsIdFile(tsId, MRC)
            self._convertOrLinkVolume(tomo, inTomoName)

            #  Defocus info:
            # "defocus1", "defocus2", "astigmatism", "phase_shift", "defocus_mean"
            logger.info(cyanStr(f'tsId = {tsId}: generating the wedge list file...'))
            nImgs = len(presentAcqOrders)
            defocusData = np.zeros((nImgs, 5))
            counter = 0
            for ctfTomo in ctf.iterItems(orderBy=[CTFTomo.INDEX_FIELD], direction='ASC'):
                if ctfTomo.getAcquisitionOrder() in presentAcqOrders:
                    defocusData[counter, 0] = ctfTomo.getDefocusU()
                    defocusData[counter, 1] = ctfTomo.getDefocusV()
                    defocusData[counter, 2] = ctfTomo.getDefocusAngle()
                    defocusData[counter, 4] = (ctfTomo.getDefocusU() + ctfTomo.getDefocusV()) / 2
                    counter += 1

            # Tilt angles and dose
            inTltName = self._getWorkingTsIdFile(tsId, TLT)
            ts.generateTltFile(inTltName, presentAcqOrders=presentAcqOrders, includeDose=True)
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
            logger.info(cyanStr(f'tsId = {tsId}: generating the tm_params.star file...'))
            self._createTmParamsFile(tsId, tomoObjId)
        except Exception as e:
            self.failedTsIds.append(tsId)
            logger.error(redStr(f'tsId = {tsId} -> input conversion failed with the exception -> {e}'))
            logger.error(traceback.format_exc())

    def templateMatchingStep(self, tsId: str):
        if tsId in self.failedTsIds:
            return
        try:
            logger.info(cyanStr(f'===> tsId = {tsId}: performing the template matching...'))
            args = 'run_tm '
            args += f'-n {self.nTiles.get()} '
            args += f'{self._genTmParamFileName(tsId)}'
            Plugin.runGapStop(self, self.program, args)
        except Exception as e:
            self.failedTsIds.append(tsId)
            logger.error(redStr(f'tsId = {tsId} -> gapStopTM execution failed with the exception -> {e}'))
            logger.error(traceback.format_exc())

    def createOutputStep(self, tsId: str):
        if tsId in self.failedTsIds:
            return
        try:
            with self._lock:
                tomo = self.tomoDict[tsId]
                convertedOrLinkedTomoFile = self._getWorkingTsIdFile(tsId, MRC)
                tomoNum = tomo.getObjId()
                scoreTomoSet = self.createOutputSet()
                # Create the corresponding scoreTomo
                scoreTomo = GapStopScoreTomogram()
                scoresMap = self._getResultsFile(tsId, self._getResultsBName(SCORES, tomoNum))
                anglesMap = self._getResultsFile(tsId, self._getResultsBName(ANGLES, tomoNum))
                anglesList = self._getAngleFile()
                setMRCSamplingRate(scoresMap, tomo.getSamplingRate())  # Update the apix value in file header
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
        except Exception as e:
            logger.error(redStr(f'tsId = {tsId} -> Unable to register the output with exception {e}. Skipping... '))
            logger.error(traceback.format_exc())

    def closeOutputSetStep(self):
        scoreTomoSet = getattr(self, self._possibleOutputs.scoreTomogrmas.name, None)
        if scoreTomoSet:
            self._closeOutputSet()
        else:
            raise Exception('No gapStopTM scored tomograms were generated. Maybe the tomograms are too large '
                            'for the GPU/s used. Consider to bin them before and/or introduce a higher number in '
                            'the parameter "No. tiles to discompose the tomogram".')
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

    def _getCustomAngleFile(self) -> str:
        """Returns the path for custom angle file when using custom angular sampling."""
        return self._getExtraPath(f'custom_angles_{self._getSymmetry()}.txt')

    def _getAngleFile(self) -> str:
        """Returns the appropriate angle file path based on whether custom sampling is used."""
        if self.useCustomAngSampling.get():
            return self._getCustomAngleFile()
        else:
            return self._getCryoCatAngleFile()

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
                join('..', basename(self._getAngleFile())),  # anglist_name
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
        inVol = self._getFormAttrib(REF_VOL)
        inMask = self._getFormAttrib(IN_MASK)
        
        if not tsSet:
            tsSetRel = self._getTsFromRelations()
            if not tsSetRel or not self.currentBin.get():
                valMsg.append('Unable to find via relations the tilt-series corresponding to the '
                              'introduced tomograms. Please introduce them manually (advanced parameters).')
        if np.abs(inVol.getSamplingRate() - inMask.getSamplingRate())>0.1:
            valMsg.append('The template and the mask do not present the same pixel size. Please resample or'
                          'resize one of them.')
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

