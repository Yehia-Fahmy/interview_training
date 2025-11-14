"""
Exercise 5: INT8 Quantization

Objective: Implement post-training INT8 quantization and understand accuracy vs speed trade-offs.

Tasks:
1. Load a pre-trained model
2. Prepare calibration data
3. Apply INT8 quantization
4. Compare accuracy and speed
"""

import torch
import torch.nn as nn
from torch.quantization import quantize_dynamic, quantize_fx
from typing import List, Tuple


class SimpleClassifier(nn.Module):
    """Simple classifier for quantization testing"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.features(x)


def evaluate_model(model: nn.Module, test_loader, device: torch.device) -> float:
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return 100.0 * correct / total


def profile_inference_time(model: nn.Module, input_tensor: torch.Tensor, num_runs: int = 100) -> float:
    """Profile inference time"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    import time
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    return (time.time() - start) / num_runs * 1000  # ms


def main():
    """Main function to run the exercise"""
    print("=" * 60)
    print("Exercise 5: INT8 Quantization")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # TODO: Load or create a pre-trained model
    # For this exercise, we'll create a simple model
    model = SimpleClassifier().to(device)
    model.eval()
    
    # Create dummy data for testing
    dummy_input = torch.randn(1, 28, 28).to(device)
    
    print("\n1. Profiling original FP32 model...")
    fp32_time = profile_inference_time(model, dummy_input)
    print(f"   Average inference time: {fp32_time:.3f} ms")
    
    # TODO: Apply dynamic quantization
    # Hint: Use torch.quantization.quantize_dynamic()
    # For linear layers: quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    
    print("\n2. Applying INT8 quantization...")
    # quantized_model = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    
    # TODO: Profile quantized model
    # quantized_time = profile_inference_time(quantized_model, dummy_input)
    # print(f"   Average inference time: {quantized_time:.3f} ms")
    # print(f"   Speedup: {fp32_time / quantized_time:.2f}x")
    
    # TODO: Compare accuracy (if you have test data)
    # fp32_acc = evaluate_model(model, test_loader, device)
    # quantized_acc = evaluate_model(quantized_model, test_loader, device)
    # print(f"   FP32 Accuracy: {fp32_acc:.2f}%")
    # print(f"   INT8 Accuracy: {quantized_acc:.2f}%")
    # print(f"   Accuracy drop: {fp32_acc - quantized_acc:.2f}%")
    
    print("\n" + "=" * 60)
    print("Exercise complete! Implement the TODOs above.")
    print("=" * 60)
    print("\nNote: For FX Graph Mode quantization, use quantize_fx()")
    print("This requires calibration data and is more accurate.")


if __name__ == "__main__":
    main()

