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
