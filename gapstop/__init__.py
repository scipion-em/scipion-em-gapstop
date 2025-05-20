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
import os
from os.path import join

import pwem
from gapstop.constants import GAPSTOP_ENV_ACTIVATION, GAPSTOP_DEFAULT_ACTIVATION_CMD, GAPSTOP, \
    GAPSTOP_CUDA_LIB, GAPSTOP_HOME, GAPSTOP_ENV_NAME, CRYOCAT, CRYOCAT_DEFAULT_VERSION, GAPSTOP_DEFAULT_VERSION, \
    GAPSTOP_03
from pyworkflow.utils import Environ

__version__ = '3.0.0'
_logo = "icon.png"
_references = ['CruzLeon2024']


class Plugin(pwem.Plugin):
    _pathVars = [GAPSTOP_CUDA_LIB]
    _supportedVersions = [GAPSTOP_03]
    _url = "https://github.com/scipion-em/scipion-em-gapstop"

    @classmethod
    def _defineVariables(cls):
        cls._defineVar(GAPSTOP_ENV_ACTIVATION, GAPSTOP_DEFAULT_ACTIVATION_CMD)
        cls._defineVar(GAPSTOP_CUDA_LIB, pwem.Config.CUDA_LIB)
        cls._defineEmVar(GAPSTOP_HOME, GAPSTOP + '-' + GAPSTOP_DEFAULT_VERSION)
        
    @classmethod
    def getGapStopEnvActivation(cls):
        return cls.getVar(GAPSTOP_ENV_ACTIVATION)

    @classmethod
    def getEnviron(cls):
        """ Setup the environment variables needed to launch gapstop. """
        environ = Environ(os.environ)
        if 'PYTHONPATH' in environ:
            # this is required for python virtual env to work
            del environ['PYTHONPATH']
        cudaLib = cls.getVar(GAPSTOP_CUDA_LIB, pwem.Config.CUDA_LIB)
        environ.addLibrary(cudaLib)

    @classmethod
    def defineBinaries(cls, env):
        CONDA_ENV_INSTALLED = 'conda_env_installed'
        CRYOCAT_INSTALLED = f'{CRYOCAT}_installed'
        GAPSTOP_INSTALLED = f'{GAPSTOP}_installed'
        gapStopHome = join(pwem.Config.EM_ROOT, GAPSTOP + '-' + GAPSTOP_DEFAULT_VERSION)
        cryoCatHome = join(gapStopHome, CRYOCAT)
        gapStopHomeTm = join(gapStopHome, GAPSTOP)

        # Create the environment and activate the conda environment
        condaEnvCmd = cls.getCondaActivationCmd()
        condaEnvCmd += f' conda create -y -n {GAPSTOP_ENV_NAME} -c conda-forge python=3.10 '
        condaEnvCmd += 'mpi4py '
        condaEnvCmd += 'jax '
        condaEnvCmd += '"jaxlib=*=*cuda*" jax && '
        condaEnvCmd += f'conda activate {GAPSTOP_ENV_NAME} && '
        condaEnvCmd += f'touch {CONDA_ENV_INSTALLED}'

        # Install cryoCAT (the latest tagged update is not present in Pypi, so we clone it)
        cryoCatCmd = 'git clone https://github.com/turonova/cryoCAT.git && '
        cryoCatCmd += f'cd {cryoCatHome} && '
        cryoCatCmd += f'git checkout tags/{CRYOCAT_DEFAULT_VERSION} && '
        cryoCatCmd += cls.getCondaActivationCmd()
        cryoCatCmd += f'conda activate {GAPSTOP_ENV_NAME} && '
        cryoCatCmd += f'pip install -e {cryoCatHome} && '
        cryoCatCmd += 'cd .. && '
        cryoCatCmd += f'touch {CRYOCAT_INSTALLED}'

        # Install GapStop_TM (this package is neither in PyPi, so we also clone this one)
        gapStopCmd = 'git clone https://gitlab.mpcdf.mpg.de/bturo/gapstop_tm.git && '
        gapStopCmd += f'cd {gapStopHomeTm} && '
        gapStopCmd += f'git checkout tags/{GAPSTOP_DEFAULT_VERSION} && '
        gapStopCmd += cls.getCondaActivationCmd()
        gapStopCmd += f'conda activate {GAPSTOP_ENV_NAME} && '
        gapStopCmd += f'pip install -e {gapStopHomeTm} && '
        gapStopCmd += 'cd .. && '
        gapStopCmd += f'touch {GAPSTOP_INSTALLED}'

        installationCmds = [(condaEnvCmd, CONDA_ENV_INSTALLED),
                            (cryoCatCmd, CRYOCAT_INSTALLED),
                            (gapStopCmd, GAPSTOP_INSTALLED)]

        envPath = os.environ.get('PATH', "")  # keep path since conda likely in there
        installEnvVars = {'PATH': envPath} if envPath else None

        env.addPackage(GAPSTOP,
                       version=GAPSTOP_DEFAULT_VERSION,
                       tar='void.tgz',
                       commands=installationCmds,
                       neededProgs=cls.getDependencies(),
                       vars=installEnvVars,
                       default=True)

    @classmethod
    def getDependencies(cls):
        # try to get CONDA activation command
        condaActivationCmd = cls.getCondaActivationCmd()
        neededProgs = ['git']
        if not condaActivationCmd:
            neededProgs.append('conda')
        return neededProgs

    @classmethod
    def runGapStop(cls, protocol, program, args, cwd=None, numberOfMpi=1, isCryoCatExec=False):
        """ Run gapstop command from a given protocol. """
        cmd = cls.getCondaActivationCmd() + " "
        cmd += cls.getGapStopEnvActivation()
        if isCryoCatExec:
            cmd += f" && {program} "
        else:
            cmd += f" && CUDA_VISIBLE_DEVICES=%(GPU)s {program} "

        protocol.runJob(cmd, args, env=cls.getEnviron(), cwd=cwd, numberOfMpi=numberOfMpi)


