from setuptools import setup, find_packages

setup(
    name="rl_pendulum",
    version="1.0.0",
    description="Sim-to-Real RL for inverted pendulum balance on ESP32",
    author="RL-Pendulum Contributors",
    python_requires=">=3.10",
    packages=find_packages(exclude=["tests*", "firmware*"]),
    install_requires=[
        "gymnasium>=0.29.0",
        "stable-baselines3>=2.3.0",
        "torch>=2.2.0",
        "numpy>=1.26.0",
        "onnx>=1.16.0",
        "onnxruntime>=1.18.0",
        "matplotlib>=3.8.0",
        "pyyaml>=6.0.1",
        "tqdm>=4.66.0",
        "scipy>=1.13.0",
    ],
    entry_points={
        "console_scripts": [
            "rl-pendulum-train=training.train:main",
            "rl-pendulum-eval=evaluation.evaluate:main",
            "rl-pendulum-export=export.export_onnx:main",
        ]
    },
)
