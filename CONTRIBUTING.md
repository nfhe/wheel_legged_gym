# 贡献指南

感谢您对 Wheeled Bipedal Gym 项目的关注！我们欢迎所有形式的贡献，包括但不限于：

- 🐛 Bug 报告
- 💡 功能建议
- 📝 文档改进
- 🔧 代码贡献
- 🧪 测试用例

## 开发环境设置

1. **Fork 并克隆仓库**
```bash
git clone https://github.com/YOUR_USERNAME/wheel_legged_gym.git
cd wheel_legged_gym
```

2. **创建 Conda 环境**
```bash
conda create -n wheeled_bipedal_dev python=3.8
conda activate wheeled_bipedal_dev
```

3. **安装开发依赖**
```bash
pip install -e .
```

## 代码规范

### Python 代码风格
- 遵循 [PEP 8](https://www.python.org/dev/peps/pep-0008/) 代码风格
- 使用 [Black](https://black.readthedocs.io/) 进行代码格式化
- 行长度限制：88 字符

### 提交信息规范
使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

类型包括：
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档更新
- `style`: 代码格式调整
- `refactor`: 代码重构
- `test`: 测试相关
- `chore`: 构建过程或辅助工具的变动

### 示例
```
feat(env): 添加新的地形类型支持

- 新增斜坡地形
- 优化地形生成算法
- 更新相关测试用例

Closes #123
```

## 提交流程

1. **创建功能分支**
```bash
git checkout -b feature/your-feature-name
```

2. **进行更改并测试**
```bash
# 运行测试
pytest tests/

# 检查代码风格
black --check .
flake8 .
```

3. **提交更改**
```bash
git add .
git commit -m "feat: 添加新功能描述"
```

4. **推送到远程仓库**
```bash
git push origin feature/your-feature-name
```

5. **创建 Pull Request**
   - 在 GitHub 上创建 Pull Request
   - 填写详细的描述
   - 关联相关 Issue

## 测试指南

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_env.py

# 运行带覆盖率的测试
pytest --cov=wheeled_bipedal_gym
```

### 添加新测试
- 测试文件应放在 `tests/` 目录下
- 测试函数名以 `test_` 开头
- 使用描述性的测试名称

## 文档贡献

### 更新 README
- 保持文档与代码同步
- 使用清晰的示例
- 添加必要的截图或 GIF

### API 文档
- 为所有公共函数添加文档字符串
- 使用 Google 或 NumPy 风格的文档字符串
- 包含参数类型和返回值说明

## 问题报告

报告 Bug 时，请包含以下信息：

1. **环境信息**
   - 操作系统版本 (Ubuntu 20.04)
   - Python 版本
   - Isaac Gym 版本
   - GPU 型号（如果使用）
   - Conda 环境信息

2. **错误描述**
   - 详细的错误信息
   - 重现步骤
   - 预期行为

3. **附加信息**
   - 错误截图
   - 日志文件
   - 相关代码片段

## 功能建议

提出新功能建议时，请考虑：

1. **功能价值**
   - 解决的问题
   - 目标用户
   - 使用场景

2. **实现可行性**
   - 技术复杂度
   - 依赖关系
   - 维护成本

3. **设计细节**
   - API 设计
   - 配置选项
   - 向后兼容性

## 社区准则

- 尊重所有贡献者
- 保持专业和友善的交流
- 提供建设性的反馈
- 帮助新贡献者

## 联系方式

如有问题或建议，请通过以下方式联系：

- 🐛 Issues：[GitHub Issues](https://github.com/nfhe/wheel_legged_gym/issues)
- 💬 讨论：[GitHub Discussions](https://github.com/nfhe/wheel_legged_gym/discussions)

感谢您的贡献！ 🙏 