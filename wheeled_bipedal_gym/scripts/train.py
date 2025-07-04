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

import numpy as np
import os
from datetime import datetime

import isaacgym
from wheeled_bipedal_gym.envs import *
from wheeled_bipedal_gym.utils import get_args, task_registry
import torch

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    task_registry.save_cfgs(name=args.task)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations,init_at_random_ep_len=True)

if __name__ == "__main__":
    args = get_args()
    args.task = "balio_vmc"
    args.max_iterations = 6000
    # args.num_envs = 2
    # args.headless = True
    # args.resume = True
    # args.load_run = "Nov21_11-32-39_"
    # args.checkpoint = 2000
    train(args)
    #  tensorboard --logdir logs/balio_vmc/Nov20_16-42-20_
    #  tensorboard --logdir /home/he/quad/wheel_legged/lsy/v2/wheeled_bipedal_gym/logs/balio_vmc/Oct21_16-20-27_

    # 继续训练
    # /home/he/quad/wheel_legged/lsy/v6/wheeled_bipedal_gym/wheeled_bipedal_gym/scripts
    # python train.py --resume --load_run=Jun18_23-26-16_ --checkpoint=500

    # logs
    # /home/he/quad/wheel_legged/lsy/v6/wheeled_bipedal_gym
    # tensorboard --logdir=logs






