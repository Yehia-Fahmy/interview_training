"""
Challenge 1: Post-Training Quantization
Baseline Implementation - Simple, correct but unoptimized

This is a straightforward implementation that works but may not be the most efficient.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
import time


class SimpleCNN(nn.Module):
    """Simple CNN for quantization testing"""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def evaluate_accuracy(model: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Evaluate model accuracy - baseline implementation"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100.0 * correct / total


def measure_inference_time(
    model: nn.Module, 
    input_tensor: torch.Tensor, 
    num_runs: int = 100,
    warmup_runs: int = 10
) -> float:
    """Measure inference time - baseline implementation"""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Synchronize if CUDA
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    # Time multiple runs
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    elapsed = time.time() - start_time
    return (elapsed / num_runs) * 1000  # Convert to milliseconds


def measure_memory_usage(model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, float]:
    """Measure memory usage - baseline implementation"""
    if not torch.cuda.is_available():
        # CPU memory measurement is less straightforward
        model_size = sum(p.numel() * p.element_size() for p in model.parameters())
        return {
            'model_size_mb': model_size / (1024 * 1024),
            'peak_memory_mb': 0.0  # Not easily measurable on CPU
        }
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Calculate model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Run forward pass to measure peak memory
    with torch.no_grad():
        _ = model(input_tensor)
    
    peak_memory = torch.cuda.max_memory_allocated()
    
    return {
        'model_size_mb': model_size / (1024 * 1024),
        'peak_memory_mb': peak_memory / (1024 * 1024)
    }


def prepare_calibration_data(test_loader: DataLoader, num_samples: int = 100) -> List[torch.Tensor]:
    """Prepare calibration data - baseline implementation"""
    calibration_data = []
    count = 0
    
    for inputs, _ in test_loader:
        for input_tensor in inputs:
            calibration_data.append(input_tensor.unsqueeze(0))
            count += 1
            if count >= num_samples:
                return calibration_data
    
    return calibration_data


def apply_ptq(
    model: nn.Module,
    calibration_data: List[torch.Tensor],
    device: torch.device
) -> nn.Module:
    """Apply PTQ - baseline using eager mode quantization"""
    model.eval()
    model = model.to('cpu')  # Quantization typically works on CPU
    
    # Set quantization config
    model.qconfig = quant.get_default_qconfig('fbgemm')  # For x86 CPUs
    
    # Prepare model (insert observers)
    prepared_model = quant.prepare(model, inplace=False)
    
    # Calibrate
    with torch.no_grad():
        for data in calibration_data:
            data = data.to('cpu')
            _ = prepared_model(data)
    
    # Convert to quantized
    quantized_model = quant.convert(prepared_model, inplace=False)
    
    return quantized_model.to(device)


def compare_models(
    fp32_model: nn.Module,
    quantized_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Compare models - baseline implementation"""
    # Get dummy input for timing
    dummy_input = next(iter(test_loader))[0][:1].to(device)
    
    results = {}
    
    # Accuracy
    results['fp32_accuracy'] = evaluate_accuracy(fp32_model, test_loader, device)
    results['quantized_accuracy'] = evaluate_accuracy(quantized_model, test_loader, device)
    results['accuracy_drop'] = results['fp32_accuracy'] - results['quantized_accuracy']
    
    # Inference time
    results['fp32_time_ms'] = measure_inference_time(fp32_model, dummy_input)
    results['quantized_time_ms'] = measure_inference_time(quantized_model, dummy_input)
    results['speedup'] = results['fp32_time_ms'] / results['quantized_time_ms']
    
    # Memory
    fp32_memory = measure_memory_usage(fp32_model, dummy_input)
    quantized_memory = measure_memory_usage(quantized_model, dummy_input)
    results['fp32_memory_mb'] = fp32_memory['model_size_mb']
    results['quantized_memory_mb'] = quantized_memory['model_size_mb']
    results['memory_reduction'] = (
        (results['fp32_memory_mb'] - results['quantized_memory_mb']) / 
        results['fp32_memory_mb'] * 100
    )
    
    return results


def main():
    """Main function"""
    print("=" * 70)
    print("Challenge 1: Post-Training Quantization - Baseline Implementation")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleCNN(num_classes=10).to(device)
    model.eval()
    
    # Create dummy test data
    inputs = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 10, (100,))
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Prepare calibration data
    calibration_data = prepare_calibration_data(test_loader, num_samples=50)
    
    # Apply PTQ
    quantized_model = apply_ptq(model, calibration_data, device)
    
    # Compare
    results = compare_models(model, quantized_model, test_loader, device)
    
    print("\nResults:")
    print(f"  Accuracy: FP32={results['fp32_accuracy']:.2f}%, "
          f"INT8={results['quantized_accuracy']:.2f}%, "
          f"Drop={results['accuracy_drop']:.2f}%")
    print(f"  Speed: FP32={results['fp32_time_ms']:.3f}ms, "
          f"INT8={results['quantized_time_ms']:.3f}ms, "
          f"Speedup={results['speedup']:.2f}x")
    print(f"  Memory: FP32={results['fp32_memory_mb']:.2f}MB, "
          f"INT8={results['quantized_memory_mb']:.2f}MB, "
          f"Reduction={results['memory_reduction']:.1f}%")


if __name__ == "__main__":
    main()

