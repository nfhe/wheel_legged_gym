import onnx
import numpy as np

# onnx_model = onnx.load("/home/he/quad/wheel_legged/lsy/v6/wheeled_bipedal_gym/logs/balio_vmc/exported/policies/actor_model.onnx")
onnx_model = onnx.load("/home/he/quad/wheel_legged/lsy/v7/wheeled_bipedal_gym/logs/balio_vmc/exported/policies/actor_model.onnx")
onnx_weights = {}

for tensor in onnx_model.graph.initializer:
    onnx_weights[tensor.name] = np.frombuffer(tensor.raw_data, dtype=np.float32).reshape(tuple(tensor.dims))

# 保存权重以便后续比较
np.save("onnx_weights.npy", onnx_weights)
