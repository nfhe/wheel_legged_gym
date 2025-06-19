import onnx
# 验证是否导出成功
# 读取onnx模型
onnx_model = onnx.load('/home/he/quad/wheel_legged/lsy/v7/wheeled_bipedal_gym/logs/balio_vmc/exported/policies/actor_model.onnx')
# onnx_model = onnx.load('/home/he/quad/wheel_legged/lsy/v6/wheeled_bipedal_gym/logs/balio_vmc/exported/policies/model.onnx')
# 检查模型格式是否正确
onnx.checker.check_model(onnx_model)
# 以可读的形式打印计算图
print(onnx.helper.printable_graph(onnx_model.graph))
