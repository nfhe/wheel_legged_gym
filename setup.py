from setuptools import find_packages
from distutils.core import setup

setup(
    name="wheeled_bipedal_gym",
    version="1.0.0",
    author="nfhe",
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email="nfhe@example.com",
    description="高性能轮式双足机器人强化学习训练框架 - 基于 Isaac Gym 的先进仿真环境",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "isaacgym",
        "matplotlib",
        "tensorboard",
        "setuptools==59.5.0",
        "numpy<=1.20.0",
        "GitPython",
        "onnx",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
    ],
    python_requires=">=3.8",
    url="https://github.com/nfhe/wheel_legged_gym",
)
