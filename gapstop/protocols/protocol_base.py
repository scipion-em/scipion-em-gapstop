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
from os.path import join
from typing import Union

from gapstop.objects import SetOfGapStopScoreTomograms
from gapstop.constants import *
from pwem.objects import Volume
from pwem.protocols import EMProtocol
from pyworkflow.object import Pointer
from tomo.objects import SetOfTiltSeries, SetOfTomograms, SetOfCTFTomoSeries


class ProtGapStopBase(EMProtocol):
    """
    Provides the shared foundation for GAPSTOP-based cryo-electron tomography
    workflows. The protocol establishes common organizational utilities used
    throughout template matching and particle extraction procedures, enabling
    consistent management of tomographic datasets, intermediate processing
    files, and workflow-specific metadata.

    AI Generated:

    GapStop Base Protocol (ProtGapStopBase) - User Manual
        Overview

        The GapStop Base protocol acts as the common infrastructure layer for
        GAPSTOP tomography workflows inside Scipion environments. Its purpose
        is to provide a unified framework for handling tomograms, tilt-series,
        score maps, and related processing resources across multiple stages of
        cryo-electron tomography analysis.

        In biological cryo-ET workflows, large numbers of tomograms and
        associated metadata files are often processed together through complex
        multistep pipelines. Maintaining consistency between datasets,
        intermediate results, and derived outputs becomes essential for
        reproducibility and reliable structural interpretation. This protocol
        establishes the shared organizational logic that allows higher-level
        workflows to operate coherently across many tomograms and processing
        stages.

        General Role Within GAPSTOP Workflows

        Rather than performing a biological analysis directly, the protocol
        serves as the structural backbone supporting downstream template
        matching and coordinate extraction tasks. It ensures that tomographic
        data, processing products, and derived particle information remain
        consistently associated throughout the workflow.

        From a practical perspective, this organizational layer simplifies the
        handling of complex cryo-ET projects where multiple tomograms,
        intermediate maps, and metadata files must remain synchronized. This
        becomes especially important in high-throughput in situ structural
        biology studies involving many experimental conditions or large-scale
        cellular datasets.

        Dataset and Metadata Management

        Cryo-electron tomography workflows depend heavily on accurate
        relationships between tomograms, tilt-series, score maps, masks, and
        coordinate datasets. The protocol provides a common mechanism for
        retrieving and managing these interconnected biological data objects.

        This unified handling reduces the risk of inconsistencies between
        different workflow stages and helps preserve the biological context of
        each dataset. Reliable dataset organization is particularly important
        when analyzing heterogeneous cellular environments, where multiple
        structures and acquisition conditions may coexist within the same
        project.

        Tomogram-Centered Organization

        The protocol organizes processing activities around individual
        tomograms and their associated identifiers. This tomogram-centric
        structure supports scalable analysis workflows in which each tomogram
        can be processed independently while remaining integrated into the
        larger experimental dataset.

        Biologically, this organization reflects the natural structure of
        cryo-ET experiments, where each tomogram represents a distinct sampled
        cellular environment or specimen region. Preserving this separation is
        important for downstream interpretation, quality control, and
        comparative analysis between experimental conditions.

        Intermediate Processing Support

        Modern cryo-ET workflows frequently generate substantial numbers of
        intermediate files, including transformed volumes, score maps,
        particle lists, orientation information, and temporary processing
        resources. The protocol provides a consistent framework for storing
        and organizing these materials during execution.

        Proper management of intermediate data is important not only for
        computational stability but also for reproducibility. In biological
        studies, the ability to trace how coordinates or structural results
        were generated can be essential when validating discoveries or
        comparing alternative processing strategies.

        Integration Within Automated Pipelines

        The protocol is designed to operate as a reusable foundation within
        automated Scipion workflows. By centralizing common organizational
        tasks, it allows higher-level protocols to focus on biologically
        meaningful operations such as template matching, particle detection,
        and subtomogram analysis.

        This modular design supports scalable cryo-ET processing pipelines
        suitable for facility environments, collaborative projects, and
        high-throughput structural biology studies.

        Outputs and Workflow Consistency

        Although the protocol itself does not directly produce biological
        results, it plays a critical role in ensuring that downstream outputs
        remain internally consistent and properly linked to their originating
        tomograms and metadata.

        This consistency is essential for later stages such as subtomogram
        extraction, averaging, classification, and visualization. Reliable
        dataset relationships help preserve the biological meaning of detected
        structures and reduce the possibility of interpretation errors.

        Final Perspective

        In cryo-electron tomography, robust data organization is fundamental
        to reliable structural analysis. The GapStop Base protocol provides
        the common operational framework that supports complex tomography
        workflows, enabling downstream biological interpretation to proceed in
        a consistent, scalable, and reproducible manner.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # --------------------------- UTILS functions -----------------------------
    def _getFormAttrib(self, attribName: str, returnPointer: bool = False) -> Union[SetOfTiltSeries,
    SetOfTomograms, SetOfCTFTomoSeries, Volume, Pointer, SetOfGapStopScoreTomograms, None]:
        inTsPointer = getattr(self, attribName, None)
        if not inTsPointer:
            return None
        else:
            return inTsPointer if returnPointer else inTsPointer.get()

    def _getCurrentTomoDir(self, tsId: str) -> str:
        return self._getExtraPath(tsId)

    def _getTsIdExtraDirFile(self, tsId: str, fileName: str, ext: str=EM) -> str:
        return join(self._getCurrentTomoDir(tsId), fileName + ext)
