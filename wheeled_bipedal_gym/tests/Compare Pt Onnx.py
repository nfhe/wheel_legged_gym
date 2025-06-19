import torch
import onnxruntime as ort
import numpy as np

# 生成固定的27个float输入
# test_input = np.random.rand(27).astype(np.float32)

test_input = np.array([
    -0.0009, 0.0855, 0.0247, 0.0241, 0.0002, -0.9997, 0.0000, 0.0000, 0.7000, 0.0192,
    0.0054, -0.0073, 0.0119, 0.5267, 0.5260, 0.0163, 0.0022, 0.0000, 0.0000, 0.3328,
    0.3202, -0.0494, 0.0237, 0.3679, 0.0185, 0.5324, 0.2010
], dtype=np.float32)

def test_pt_model(pt_model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 选择可用的设备
    model = torch.jit.load(pt_model_path, map_location=device)  # 把模型加载到相同设备
    model.eval()

    input_tensor = torch.tensor(test_input).unsqueeze(0).to(device)  # 加载到同一设备
    with torch.no_grad():
        output = model(input_tensor)

    print("PyTorch Output:", output.cpu().squeeze().numpy())  # 确保输出在 CPU 上打印


def test_onnx_model(onnx_model_path):
    session = ort.InferenceSession(onnx_model_path)
    input_name = session.get_inputs()[0].name

    output = session.run(None, {input_name: test_input.reshape(1, -1)})
    print("ONNX Output:", output[0].squeeze())

if __name__ == "__main__":
    pt_model_path = "/home/he/quad/wheel_legged/lsy/v6/wheeled_bipedal_gym/logs/balio_vmc/exported/policies/policy_1.pt"  # 你的 PyTorch 模型路径
    onnx_model_path = "/home/he/quad/wheel_legged/lsy/v6/wheeled_bipedal_gym/logs/balio_vmc/exported/policies/actor_model.onnx"  # 你的 ONNX 模型路径
    
    print("Testing PyTorch Model...")
    test_pt_model(pt_model_path)
    
    print("\nTesting ONNX Model...")
    test_onnx_model(onnx_model_path)


