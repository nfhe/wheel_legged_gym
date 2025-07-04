# SPDX-FileCopyrightText: Copyright (c) 2024 nfhe. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 nfhe

from wheeled_bipedal_gym import WHEELED_BIPEDAL_GYM_ROOT_DIR, WHEELED_BIPEDAL_GYM_ENVS_DIR

from .base.wheeled_bipedal import WheeledBipedal
from .diablo.diablo_config import DiabloCfg, DiabloCfgPPO

from .diablo_vmc.diablo_vmc import DiabloVMC
from .diablo_vmc.diablo_vmc_config import DiabloVMCCfg, DiabloVMCCfgPPO

import os
from wheeled_bipedal_gym.utils.task_registry import task_registry

from .balio.balio_config import BalioCfg, BalioCfgPPO

from .balio_vmc.balio_vmc import BalioVMC
from .balio_vmc.balio_vmc_config import BalioVMCCfg, BalioVMCCfgPPO

# from .balio_vmc_advanced.balio_vmc_advanced_config import BalioVMCAdvancedCfg, BalioVMCAdvancedCfgPPO

task_registry.register("diablo", WheeledBipedal, DiabloCfg(), DiabloCfgPPO())
task_registry.register("diablo_vmc", DiabloVMC, DiabloVMCCfg(), DiabloVMCCfgPPO())
task_registry.register("balio", WheeledBipedal, BalioCfg(), BalioCfgPPO())
task_registry.register("balio_vmc", BalioVMC, BalioVMCCfg(), BalioVMCCfgPPO())
# task_registry.register("balio_vmc_advanced", BalioVMC, BalioVMCAdvancedCfg(), BalioVMCAdvancedCfgPPO())

