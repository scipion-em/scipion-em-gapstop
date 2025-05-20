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

GAPSTOP_HOME = 'GAPSTOP_HOME'
GAPSTOP = 'gapstop_tm'
CRYOCAT = 'cryoCAT'

# Supported versions
GAPSTOP_03 = '0.3'
GAPSTOP_DEFAULT_VERSION = GAPSTOP_03
CRYOCAT_DEFAULT_VERSION = 'v0.3.0'

GAPSTOP_ENV_NAME = f'{GAPSTOP}-{GAPSTOP_DEFAULT_VERSION}'
GAPSTOP_ENV_ACTIVATION = 'GAPSTOP_ENV_ACTIVATION'
GAPSTOP_DEFAULT_ACTIVATION_CMD = 'conda activate %s' % GAPSTOP_ENV_NAME
GAPSTOP_CUDA_LIB = 'GAPSTOP_CUDA_LIB'

# Inputs
IN_TOMOS = 'inTomos'
IN_CTF_SET = 'inCtfSet'
IN_TS_SET = 'inTsSet'
REF_VOL = 'reference'
IN_MASK = 'mask'
IN_SCORE_TOMOS = 'inScoreTomos'
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