import numpy as np

pt_weights = np.load("/home/he/quad/wheel_legged/lsy/v6/pt_weights.npy", allow_pickle=True).item()
onnx_weights = np.load("/home/he/quad/wheel_legged/lsy/v6/onnx_weights.npy", allow_pickle=True).item()

for name in pt_weights:
    if name in onnx_weights:
        diff = np.abs(pt_weights[name] - onnx_weights[name]).max()
        print(f"Layer {name}: Max difference = {diff}")
    else:
        print(f"Layer {name} is missing in ONNX model.")
