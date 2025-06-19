from wheeled_bipedal_gym import WHEELED_BIPEDAL_GYM_ROOT_DIR
import cv2
import os

import isaacgym
from isaacgym.torch_utils import *
from wheeled_bipedal_gym.envs import *
from wheeled_bipedal_gym.utils import get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch
import pandas as pd
# new imports
from PIL import Image as im
from wheeled_bipedal_gym.utils.helpers import class_to_dict
import re
from wheeled_bipedal_gym.rsl_rl.modules.actor_critic import ActorCritic

def find_latest_model_file(train_cfg):
    # 构建 logs 目录的完整路径
    logs_dir = os.path.join(WHEELED_BIPEDAL_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name)

    # 找到日期最新的文件夹，排除"exported"文件夹
    date_folders = [f for f in os.listdir(logs_dir) if os.path.isdir(os.path.join(logs_dir, f)) and f != "exported"]
    date_folders.sort(reverse=True)
    latest_date_folder = os.path.join(logs_dir, date_folders[0]) if date_folders else None

    if latest_date_folder:
        # 找到 model_xxxxx.pt 文件名里 xxxxx 数值最大的 .pt 文件
        model_files = []
        for f in os.listdir(latest_date_folder):
            match = re.match(r'model_(\d+).pt', f)
            if match:
                num = int(match.group(1))
                model_files.append((num, f))
        if model_files:
            model_files.sort(reverse=True)
            latest_model_file = os.path.join(latest_date_folder, model_files[0][1])
            try:
                # 加载模型
                model_dict = torch.load(latest_model_file)
                return model_dict
            except Exception as e:
                print(f"加载模型时出现错误: {e}")
        else:
            print("在最新日期文件夹中未找到符合条件的模型文件。")
    else:
        print("未找到日期文件夹。")
    return None

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    # env_cfg.env.episode_length_s = 20
    # env_cfg.env.fail_to_terminal_time_s = 3
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor_torque = False
    env_cfg.domain_rand.randomize_action_delay = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_Kp = False
    env_cfg.domain_rand.randomize_Kd = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_default_dof_pos = False
    env_cfg.terrain.max_init_terrain_level = env_cfg.terrain.num_rows - 1
    env_cfg.terrain.curriculum = True
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()[0]

    # load policy partial_checkpoint_load
    policy_cfg_dict = class_to_dict(train_cfg.policy)
    runner_cfg_dict = class_to_dict(train_cfg.runner)
    actor_critic_class = eval(runner_cfg_dict["policy_class_name"])
    policy: ActorCritic = actor_critic_class(env.num_obs,
                                             env.num_privileged_obs,
                                             env.num_actions,
                                                **policy_cfg_dict)
    print(policy)
    if EXPORT_POLICY:
        path = os.path.join(WHEELED_BIPEDAL_GYM_ROOT_DIR, "logs", train_cfg.runner.experiment_name,
            "exported", "policies")
    model_dict = find_latest_model_file(train_cfg)
    policy.load_state_dict(model_dict['model_state_dict'])
    # policy.half()
    policy.eval()
    policy = policy.to(env.device)
    policy.save_torch_jit_policy(path, device=env.device)
    policy.save_torch_onnx_policy(path, device=env.device)

    logger = Logger(env.dt)
    robot_index = 21  # which robot is used for logging
    joint_index = 1  # which joint is used for logging
    stop_state_log = 1000  # number of steps before plotting states
    stop_rew_log = (env.max_episode_length + 1)  # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1.0, 1.0, 0.0])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0
    latent = None

    CoM_offset_compensate = True
    vel_err_intergral = torch.zeros(env.num_envs, device=env.device)
    vel_cmd = torch.zeros(env.num_envs, device=env.device)

    for i in range(5000 * int(env.max_episode_length)):

        actions = policy.act_inference(obs)

        env.commands[:, 0] = 1.0
        env.commands[:, 2] = 0.17  # + 0.07 * np.sin(i * 0.01)
        env.commands[:, 3] = 0

        if CoM_offset_compensate:
            if i > 200 and i < 600:
                vel_cmd[:] = 2.0 * np.clip((i - 200) * 0.05, 0, 1)
            else:
                vel_cmd[:] = 0
            vel_err_intergral += (
                (vel_cmd - env.base_lin_vel[:, 0])
                * env.dt
                * ((vel_cmd - env.base_lin_vel[:, 0]).abs() < 0.5)
            )
            vel_err_intergral = torch.clip(vel_err_intergral, -0.5, 0.5)
            env.commands[:, 0] = vel_cmd + vel_err_intergral

        obs, _, rews, dones, infos, obs_history = env.step(actions)


if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = False
    MOVE_CAMERA = False
    args = get_args()
    args.task = "balio_vmc"
    # args.headless = True
    args.num_envs = 50
    play(args)






























