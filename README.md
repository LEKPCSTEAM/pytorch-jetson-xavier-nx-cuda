# YOLOv8 Jetson Xavier NX with PyTorch CUDA

This project runs YOLOv8 on NVIDIA Jetson Xavier NX using custom PyTorch and TorchVision builds optimized with CUDA.

## 🧰 Requirements

- Jetson Xavier NX (JetPack 5.x)
- Python 3.8
- [`uv`](https://github.com/astral-sh/uv) (fast Python package manager)

## 📦 Setup

### 1. Initialize Project

```bash
uv init
```

### 2. Install Dependencies

Install PyTorch and TorchVision `.whl` packages optimized for Jetson:

```bash
uv pip install torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
uv pip install torchvision-0.16.1+fdea156-cp38-cp38-linux_aarch64.whl
```

> Note: These are local .whl files tracked via Git LFS.
> 

---

### 3. Run Test

You can test CUDA support with:

```bash
python test_cuda.py
```

This script should print whether CUDA is available and which GPU is used.
