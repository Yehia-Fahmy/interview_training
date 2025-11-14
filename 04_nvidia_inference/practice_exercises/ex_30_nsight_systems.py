"""
Exercise 30: Nsight Systems Profiling

Objective: Learn to use Nsight Systems for timeline profiling of inference pipelines.

Tasks:
1. Install Nsight Systems (if not already installed)
2. Profile a simple inference pipeline
3. Analyze GPU utilization
4. Identify synchronization issues

Note: Nsight Systems is a command-line and GUI tool.
This exercise shows how to generate profiling data programmatically.
"""

import torch
import torch.nn as nn
import subprocess
import os
import time


class InferencePipeline(nn.Module):
    """Simple inference pipeline for profiling"""
    
    def __init__(self):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.backbone = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.preprocess(x)
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x


def run_inference_with_profiling(model, input_tensor, output_file="profile.nsys-rep"):
    """
    Run inference and generate Nsight Systems profile.
    
    Note: This requires Nsight Systems to be installed and in PATH.
    Install from: https://developer.nvidia.com/nsight-systems
    """
    if not torch.cuda.is_available():
        print("CUDA not available. Nsight Systems requires CUDA.")
        return
    
    # Check if nsys is available
    try:
        result = subprocess.run(['nsys', '--version'], 
                              capture_output=True, text=True, timeout=5)
        print(f"Nsight Systems version: {result.stdout.strip()}")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("Nsight Systems (nsys) not found in PATH.")
        print("Install from: https://developer.nvidia.com/nsight-systems")
        print("Or use: conda install -c nvidia nsight-systems")
        return
    
    model.eval()
    model = model.cuda()
    input_tensor = input_tensor.cuda()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    torch.cuda.synchronize()
    
    # Profile with nsys
    print(f"\nProfiling inference pipeline...")
    print(f"Output file: {output_file}")
    
    # Build nsys command
    # Note: In practice, you'd run this from command line:
    # nsys profile --output=profile.nsys-rep python script.py
    
    # For this exercise, we'll show the command and run inference
    print("\nTo profile with Nsight Systems, run:")
    print(f"  nsys profile --output={output_file} python {__file__}")
    print("\nOr use the Python API (if available):")
    
    # Run inference (without nsys wrapper for this example)
    start = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"\nInference completed:")
    print(f"  100 iterations in {elapsed:.3f} seconds")
    print(f"  Average: {elapsed / 100 * 1000:.3f} ms per inference")
    
    return output_file


def analyze_profile(profile_file):
    """
    Analyze Nsight Systems profile.
    
    Note: This requires nsys-cli or GUI tool.
    """
    if not os.path.exists(profile_file):
        print(f"Profile file not found: {profile_file}")
        return
    
    print(f"\nAnalyzing profile: {profile_file}")
    print("\nTo view the profile:")
    print(f"  1. GUI: nsys-ui {profile_file}")
    print(f"  2. CLI: nsys stats {profile_file}")
    print(f"  3. Export: nsys export --type=sqlite {profile_file}")
    
    # Try to get basic stats
    try:
        result = subprocess.run(['nsys', 'stats', profile_file],
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("\nProfile statistics:")
            print(result.stdout)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("Could not generate statistics. Use nsys-ui to view profile.")


def main():
    """Main function to run the exercise"""
    print("=" * 60)
    print("Exercise 30: Nsight Systems Profiling")
    print("=" * 60)
    
    # Create model and input
    model = InferencePipeline()
    input_tensor = torch.randn(4, 3, 224, 224)
    
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Input shape: {input_tensor.shape}")
    
    # Run inference with profiling
    profile_file = run_inference_with_profiling(model, input_tensor)
    
    # Analyze profile
    if profile_file:
        analyze_profile(profile_file)
    
    print("\n" + "=" * 60)
    print("Exercise complete!")
    print("=" * 60)
    print("\nKey metrics to look for in Nsight Systems:")
    print("  - GPU utilization (should be high)")
    print("  - Kernel execution time")
    print("  - Memory transfers")
    print("  - CPU-GPU synchronization points")
    print("  - Idle time between kernels")


if __name__ == "__main__":
    main()

