# Copyright 2024 nfhe

from wheeled_bipedal_gym.envs.balio.balio_config import (BalioCfg, BalioCfgPPO)


class BalioVMCP3O(BalioCfg):
    """P3O算法的环境配置类"""
    class env(BalioCfg.env):
        num_privileged_obs = (BalioCfg.env.num_observations + 7 * 11 + 3 + 6 * 7 + 3 + 3)
        fail_to_terminal_time_s = 1
        episode_length_s = 20

    class terrain(BalioCfg.terrain):
        mesh_type = "plane"
        # mesh_type = "trimesh"  # "heightfield" # none, plane, heightfield or trimesh
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1
        dynamic_friction = 0.
        restitution = 0.5
        # rough terrain only:
        measure_heights = True
        measured_points_x = [-0.5, -0.4,-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5] # 0.6mx1.0m rectangle (without center line)
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        selected = True  # select a unique terrain type and pass all arguments
        terrain_kwargs = None  # Dict of arguments for selected terrain
        max_init_terrain_level = 0  # starting curriculum state
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        # trimesh only:
        slope_treshold = 0.75  # slopes above this threshold will be corrected to vertical surfaces

    class rewards(BalioCfg.rewards):
        class scales:
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1
            tracking_ang_vel = 1.0

            base_height = 5 #20
            nominal_state = -0.1
            lin_vel_z = -0.1e-3  # -0.1e-3 #-2.0
            ang_vel_xy = -0.05
            orientation = -100.0

            dof_vel = -5e-2
            dof_acc = -2.5e-3
            torques = -0.1e-5 #-0.01
            action_rate = -0.01
            action_smooth = -0.01

            collision = -1.0
            dof_pos_limits = -1

            theta_limit = -0.1e-5 # -0.02
            same_l = -0.1e-8 #-0.1e-8
            same_theta = 0.5 #0.01

        base_height_target = 0.14


    class control(BalioCfg.control):
        action_scale_theta = 0.5
        action_scale_l0 = 0.1
        action_scale_vel = 10.0

        l0_offset = 0.14
        feedforward_force = 17.5  # 40[N]

        # real max
        kp_theta = 5  # [N*m/rad]
        kd_theta = 0.5  # [N*m*s/rad]
        kp_l0 = 150  # [N/m]
        kd_l0 = 4  # [N*s/m]

        # PD Drive parameters:
        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0.4}  # [N*m/rad]
        damping = {"f0": 0.0, "f1": 0.0, "wheel": 0.4}  # [N*m*s/rad]

    class normalization(BalioCfg.normalization):
        class obs_scales(BalioCfg.normalization.obs_scales):
            l0 = 5.0
            l0_dot = 0.25
            # wheel pos should be zero!
            wheel_pos = 0.0
            dof_pos = 1.0

    class noise(BalioCfg.noise):
        class noise_scales(BalioCfg.noise.noise_scales):
            l0 = 0.02
            l0_dot = 0.1

    class commands(BalioCfg.commands):
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-3.14, 3.14]  # min max [rad/s]
            height = [0.10, 0.20]
            heading = [-3.14, 3.14]

    class domain_rand(BalioCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-1.8, 1.8] # kg
        randomize_inertia = True
        randomize_inertia_range = [0.8, 1.2]
        randomize_base_com = True
        rand_com_vec = [0.05, 0.05, 0.05]
        push_robots = True
        push_interval_s = 7
        max_push_vel_xy = 2.0
        randomize_Kp = True
        randomize_Kp_range = [0.8, 1.2]
        randomize_Kd = True
        randomize_Kd_range = [0.8, 1.2]
        randomize_motor_torque = True
        randomize_motor_torque_range = [0.8, 1.2]
        randomize_default_dof_pos = True
        randomize_default_dof_pos_range = [-0.05, 0.05]
        randomize_action_delay = True
        delay_ms_range = [0, 10]

    class costs:
        class scales:
            pos_limit = 0.1
            torque_limit = 0.1
            dof_vel_limits = 0.1
        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0

    class cost:
        num_costs = 3

class BalioVMCCfgP3O(BalioCfgPPO):
    """P3O算法的训练配置类"""
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy(BalioCfgPPO.policy):
        # 继承PPO策略配置，可以根据P3O的特点进行调整
        init_noise_std = 0.5
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]
        activation = "elu"  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

        # only for ActorCriticSequence
        num_encoder_obs = (BalioVMCP3O.env.obs_history_length * BalioVMCP3O.env.num_observations)
        latent_dim = 3  # at least 3 to estimate base linear velocity
        encoder_hidden_dims = [128, 64]

    class algorithm(BalioCfgPPO.algorithm):
        # P3O特定的算法参数
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        learning_rate = 1.0e-3  # 5.e-4
        schedule = "adaptive"  # could be adaptive, fixed
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.005
        max_grad_norm = 1.0

        extra_learning_rate = 1e-3

        # P3O特定参数
        p3o_alpha = 0.1  # P3O的alpha参数，控制策略更新的保守程度
        p3o_beta = 0.5   # P3O的beta参数，控制价值函数更新的保守程度
        p3o_gamma = 0.95 # P3O的gamma参数，折扣因子

    class runner(BalioCfgPPO.runner):
        # 使用P3O算法
        policy_class_name = "ActorCritic"  # could be ActorCritic, ActorCriticSequence
        algorithm_class_name = "P3O"  # 使用P3O算法
        num_steps_per_env = 48  # per iteration
        max_iterations = 50000  # number of policy updates

        # logging
        save_interval = 100  # check for potential saves every this many iterations
        experiment_name = "balio_vmc_p3o"  # P3O实验名称
        run_name = ""
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = ''  # updated from load_run and chkpt