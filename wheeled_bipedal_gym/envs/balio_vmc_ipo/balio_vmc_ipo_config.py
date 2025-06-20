# Copyright 2024 nfhe

from wheeled_bipedal_gym.envs.balio.balio_config import (BalioCfg, BalioCfgPPO)

class BalioVMCIPO(BalioCfg):
    """IPO算法的环境配置类"""
    class env(BalioCfg.env):
        num_privileged_obs = (BalioCfg.env.num_observations + 7 * 11 + 3 + 6 * 7 + 3 + 3)
        fail_to_terminal_time_s = 1
        episode_length_s = 20
        debug_viz = True

    class terrain(BalioCfg.terrain):
        # mesh_type = "plane"
        mesh_type = "trimesh"
        horizontal_scale = 0.1  # [m]
        vertical_scale = 0.005  # [m]
        border_size = 25  # [m]
        curriculum = True
        static_friction = 1
        dynamic_friction = 0.
        restitution = 0.5
        measure_heights = True
        measured_points_x = [-0.5, -0.4,-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        measured_points_y = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        selected = True
        terrain_kwargs = None
        max_init_terrain_level = 0
        terrain_length = 8.0
        terrain_width = 8.0
        num_rows = 10
        num_cols = 20
        terrain_proportions = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        slope_treshold = 0.75

    class rewards(BalioCfg.rewards):
        class scales:
            tracking_lin_vel = 1.0
            tracking_lin_vel_enhance = 1
            tracking_ang_vel = 1.0
            base_height = 5
            nominal_state = -0.1
            lin_vel_z = -0.1e-3
            ang_vel_xy = -0.05
            orientation = -100.0
            dof_vel = -5e-2
            dof_acc = -2.5e-3
            torques = -0.1e-5
            action_rate = -0.01
            action_smooth = -0.01
            collision = -1.0
            dof_pos_limits = -1
            theta_limit = -0.1e-5
            same_l = -0.1e-8
            same_theta = 0.5
        base_height_target = 0.14

    class control(BalioCfg.control):
        action_scale_theta = 0.5
        action_scale_l0 = 0.1
        action_scale_vel = 10.0
        l0_offset = 0.14
        feedforward_force = 17.5
        kp_theta = 5
        kd_theta = 0.5
        kp_l0 = 150
        kd_l0 = 4
        stiffness = {"f0": 0.0, "f1": 0.0, "wheel": 0.4}
        damping = {"f0": 0.0, "f1": 0.0, "wheel": 0.4}

    class normalization(BalioCfg.normalization):
        class obs_scales(BalioCfg.normalization.obs_scales):
            l0 = 5.0
            l0_dot = 0.25
            wheel_pos = 0.0
            dof_pos = 1.0

    class noise(BalioCfg.noise):
        class noise_scales(BalioCfg.noise.noise_scales):
            l0 = 0.02
            l0_dot = 0.1

    class commands(BalioCfg.commands):
        class ranges:
            lin_vel_x = [-1.0, 1.0]
            ang_vel_yaw = [-3.14, 3.14]
            height = [0.10, 0.20]
            heading = [-3.14, 3.14]

    class domain_rand(BalioCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 2.0]
        randomize_restitution = True
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-1.8, 1.8]
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

class BalioVMCCfgIPO(BalioCfgPPO):
    """IPO算法的训练配置类"""
    seed = 1
    runner_class_name = "OnPolicyRunner"

    class policy(BalioCfgPPO.policy):
        init_noise_std = 0.5
        actor_hidden_dims = [128, 64, 32]
        critic_hidden_dims = [256, 128, 64]
        activation = "elu"
        num_encoder_obs = (BalioVMCIPO.env.obs_history_length * BalioVMCIPO.env.num_observations)
        latent_dim = 3
        encoder_hidden_dims = [128, 64]

    class algorithm(BalioCfgPPO.algorithm):
        # IPO特定的算法参数
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.0e-3
        schedule = "adaptive"
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.005
        max_grad_norm = 1.0
        extra_learning_rate = 1e-3
        # IPO特定参数
        ipo_alpha = 0.1  # IPO的alpha参数，控制策略更新的保守程度
        ipo_beta = 0.5   # IPO的beta参数，控制价值函数更新的保守程度
        ipo_gamma = 0.95 # IPO的gamma参数，折扣因子

    class runner(BalioCfgPPO.runner):
        policy_class_name = "ActorCritic"
        algorithm_class_name = "IPO"  # 使用IPO算法
        num_steps_per_env = 48
        max_iterations = 50000
        save_interval = 100
        experiment_name = "balio_vmc_ipo"
        run_name = ""
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = ''