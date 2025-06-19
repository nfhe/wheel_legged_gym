import torch

pt_model = torch.load("/home/he/quad/wheel_legged/lsy/v6/wheeled_bipedal_gym/logs/balio_vmc/exported/policies/actor_model.pt")  # 加载 PyTorch JIT 模型
pt_weights = {name: param.detach().cpu().numpy() for name, param in pt_model.named_parameters()}

# 保存权重以便后续比较
import numpy as np
np.save("pt_weights.npy", pt_weights)
