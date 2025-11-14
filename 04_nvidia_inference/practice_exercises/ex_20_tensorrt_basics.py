"""
Exercise 20: TensorRT Basics

Objective: Convert a PyTorch model to TensorRT and compare performance.

Tasks:
1. Load a PyTorch model
2. Convert to TensorRT
3. Compare inference speed
4. Understand optimization pipeline

Note: This exercise requires TensorRT to be installed.
Installation: Follow NVIDIA TensorRT installation guide
"""

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("TensorRT not available. Install TensorRT and pycuda.")
    print("This exercise requires:")
    print("  1. TensorRT SDK")
    print("  2. pycuda: pip install pycuda")
    print("  3. torch2trt or onnx (for conversion)")

import torch
import torch.nn as nn
import numpy as np
import time


class SimpleModel(nn.Module):
    """Simple model for TensorRT conversion"""
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def profile_pytorch_model(model, input_tensor, num_runs=100):
    """Profile PyTorch model inference"""
    model.eval()
    device = input_tensor.device
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Profile
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    return (time.time() - start) / num_runs * 1000  # ms


def main():
    """Main function to run the exercise"""
    print("=" * 60)
    print("Exercise 20: TensorRT Basics")
    print("=" * 60)
    
    if not TRT_AVAILABLE:
        print("\nTensorRT not available. Skipping exercise.")
        print("See comments at top of file for installation instructions.")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type != 'cuda':
        print("CUDA required for TensorRT. Skipping exercise.")
        return
    
    # Create model
    model = SimpleModel().to(device).eval()
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    print("\n1. Profiling PyTorch model...")
    pytorch_time = profile_pytorch_model(model, dummy_input)
    print(f"   Average inference time: {pytorch_time:.3f} ms")
    
    # TODO: Convert to TensorRT
    # There are several ways to do this:
    # 1. Using torch2trt: https://github.com/NVIDIA-AI-IOT/torch2trt
    # 2. Using ONNX: PyTorch -> ONNX -> TensorRT
    # 3. Using TensorRT Python API directly
    
    # Example with torch2trt (if installed):
    # from torch2trt import torch2trt
    # model_trt = torch2trt(model, [dummy_input], fp16_mode=True)
    
    # Example with ONNX:
    # 1. Export to ONNX: torch.onnx.export(model, dummy_input, "model.onnx")
    # 2. Use TensorRT Python API to build engine from ONNX
    
    print("\n2. Converting to TensorRT...")
    print("   TODO: Implement TensorRT conversion")
    print("   See comments above for conversion methods")
    
    # TODO: Profile TensorRT model
    # trt_time = profile_tensorrt_model(model_trt, dummy_input)
    # print(f"   Average inference time: {trt_time:.3f} ms")
    # print(f"   Speedup: {pytorch_time / trt_time:.2f}x")
    
    print("\n" + "=" * 60)
    print("Exercise complete! Implement TensorRT conversion.")
    print("=" * 60)
    print("\nNote: TensorRT conversion requires:")
    print("  - Model export (ONNX or direct)")
    print("  - Engine building with optimization")
    print("  - Runtime inference")


if __name__ == "__main__":
    main()

