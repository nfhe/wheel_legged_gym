# Wheeled Bipedal Gym

[![Python 3.8](https://img.shields.io/badge/python-3.8-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![Isaac Gym](https://img.shields.io/badge/Isaac%20Gym-Required-red.svg)](https://developer.nvidia.com/isaac-gym)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-green.svg)](LICENSE)

基于 Isaac Gym 的轮式双足机器人强化学习训练框架，支持 Balio 和 Diablo 机器人模型的高性能仿真训练。

## 🚀 项目特色

- **多机器人支持**: 支持 Balio 和 Diablo 轮式双足机器人
- **高性能训练**: 基于 Isaac Gym 的 GPU 加速并行仿真
- **先进算法**: 集成 PPO 强化学习算法
- **模块化设计**: 易于扩展和自定义的架构
- **实时监控**: 集成 TensorBoard 支持训练可视化

## 📋 目录

- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [支持的机器人](#支持的机器人)
- [训练配置](#训练配置)
- [故障排除](#故障排除)

## 🛠️ 安装指南

### 系统要求

- Ubuntu 20.04
- Python 3.8
- NVIDIA GPU (推荐 RTX 系列)
- Isaac Gym (需要 NVIDIA 开发者账号)
- Conda 环境管理

### 环境配置

1. **创建 Conda 环境**
```bash
conda create -n wheeled_bipedal python=3.8
conda activate wheeled_bipedal
```

2. **安装 PyTorch**
```bash
# 安装支持 CUDA 的 PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

3. **安装 Isaac Gym**
```bash
# 从 https://developer.nvidia.com/isaac-gym 下载 Isaac Gym Preview 4
cd isaacgym/python && pip install -e .
# 测试安装
cd examples && python 1080_balls_of_solitude.py
```

4. **安装 Wheeled Bipedal Gym**
```bash
git clone https://github.com/nfhe/wheel_legged_gym.git
cd wheeled_bipedal_gym
pip install -e .
```

## 🚀 快速开始

### 基础训练

```bash
# 激活 conda 环境
conda activate wheeled_bipedal

# 训练 Balio 机器人 (默认)
python wheeled_bipedal_gym/scripts/train.py

# 训练 Diablo 机器人
python wheeled_bipedal_gym/scripts/train.py --task=diablo_vmc
```

### 自定义训练参数

```bash
# 设置训练迭代次数
python wheeled_bipedal_gym/scripts/train.py --max_iterations 10000

# 启用无头模式（无GUI）
python wheeled_bipedal_gym/scripts/train.py --headless

# 设置并行环境数量
python wheeled_bipedal_gym/scripts/train.py --num_envs 2048
```

### 恢复训练

```bash
# 从检查点恢复训练
python wheeled_bipedal_gym/scripts/train.py --resume --load_run=YOUR_RUN_ID --checkpoint=1000
```

### 监控训练过程

```bash
# 启动 TensorBoard
tensorboard --logdir=logs

# 查看特定任务的训练日志
tensorboard --logdir=logs/balio_vmc/YOUR_RUN_ID
```

## 🤖 支持的机器人

### Balio 轮式双足机器人
- **特点**: 轻量化设计，高机动性
- **应用**: 室内导航，动态平衡控制
- **配置**: `balio_vmc`
- **收敛**: 约 3000 次迭代达到收敛

### Diablo 轮式双足机器人  
- **特点**: 高负载能力，复杂地形适应
- **应用**: 户外探索，重物搬运
- **配置**: `diablo_vmc`
- **收敛**: 约 3000 次迭代达到收敛

## ⚙️ 训练配置

### 环境参数

| 参数             | 默认值 | 描述             |
| ---------------- | ------ | ---------------- |
| `max_iterations` | 6000   | 最大训练迭代次数 |
| `num_envs`       | 4096   | 并行环境数量     |
| `headless`       | False  | 是否启用无头模式 |
| `resume`         | False  | 是否恢复训练     |

### 算法参数

- **学习率**: 自适应调整
- **批次大小**: 基于环境数量自动计算
- **折扣因子**: γ = 0.99
- **GAE 参数**: λ = 0.95

## 🔧 故障排除

### 常见问题

1. **NaN 值错误**
   ```bash
   # 降低学习率或添加梯度裁剪
   # 检查输入数据是否有异常值
   ```

2. **内存不足**
   ```bash
   # 减少并行环境数量
   python wheeled_bipedal_gym/scripts/train.py --num_envs 1024
   ```

3. **训练不稳定**
   ```bash
   # 调整奖励函数或使用更保守的学习率
   ```

4. **Conda 环境问题**
   ```bash
   # 重新创建环境
   conda deactivate
   conda env remove -n wheeled_bipedal
   conda create -n wheeled_bipedal python=3.8
   conda activate wheeled_bipedal
   ```

### 性能优化建议

- 使用 SSD 存储训练日志
- 启用 GPU 内存优化
- 调整 Isaac Gym 线程数
- 确保 conda 环境干净，避免包冲突

## 📄 许可证

本项目采用 BSD-3-Clause 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式

- **作者**: nfhe
- **项目地址**: https://github.com/nfhe/wheel_legged_gym

---

⭐ 如果这个项目对您有帮助，请给我们一个星标！

