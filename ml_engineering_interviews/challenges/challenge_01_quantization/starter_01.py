"""
Challenge 1: Post-Training Quantization
Starter Code

Implement PTQ for a pre-trained model and measure accuracy-speed trade-offs.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import DataLoader
from typing import Tuple, Dict
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
    """
    Evaluate model accuracy on test dataset.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
    
    Returns:
        Accuracy percentage
    """
    # TODO: Implement accuracy evaluation
    # Hint: Set model to eval mode, iterate through test_loader, compute accuracy
    model.eval()
    # TODO: Complete implementation
    pass


def measure_inference_time(
    model: nn.Module, 
    input_tensor: torch.Tensor, 
    num_runs: int = 100,
    warmup_runs: int = 10
) -> float:
    """
    Measure average inference time in milliseconds.
    
    Args:
        model: Model to profile
        input_tensor: Input tensor for inference
        num_runs: Number of inference runs for averaging
        warmup_runs: Number of warmup runs
    
    Returns:
        Average inference time in milliseconds
    """
    # TODO: Implement inference time measurement
    # Hint: Warmup first, then time multiple runs, average the results
    # Remember to synchronize GPU (CUDA or MPS) if using GPU
    model.eval()
    device = input_tensor.device
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Synchronize if GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    # Time multiple runs
    start_time = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    
    # Synchronize if GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()
    
    elapsed = time.perf_counter() - start_time
    return (elapsed / num_runs) * 1000  # Convert to milliseconds


def measure_memory_usage(model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Measure model memory usage in MB.
    
    Args:
        model: Model to measure
        input_tensor: Input tensor for forward pass
    
    Returns:
        Dictionary with 'model_size_mb' and 'peak_memory_mb'
    """
    # TODO: Implement memory measurement
    # Hint: Use device-specific memory APIs for GPU (CUDA or MPS)
    # Calculate model size by summing parameter sizes
    device = input_tensor.device
    
    # Calculate model size
    model_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    total_model_size = model_size + buffer_size
    
    result = {
        'model_size_mb': total_model_size / (1024 * 1024),
        'peak_memory_mb': 0.0
    }
    
    # GPU memory measurement (CUDA only, MPS doesn't expose this API)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        initial_memory = torch.cuda.memory_allocated(device)
        
        with torch.no_grad():
            _ = model(input_tensor)
        
        peak_memory = torch.cuda.max_memory_allocated(device)
        result['peak_memory_mb'] = peak_memory / (1024 * 1024)
    
    return result


def prepare_calibration_data(test_loader: DataLoader, num_samples: int = 100):
    """
    Prepare calibration data for quantization.
    
    Args:
        test_loader: DataLoader for calibration data
        num_samples: Number of samples to use for calibration
    
    Returns:
        List of input tensors for calibration
    """
    # TODO: Extract calibration data from test_loader
    # Hint: Collect input tensors (not labels) up to num_samples
    pass


def apply_ptq(
    model: nn.Module,
    calibration_data: list,
    device: torch.device
) -> nn.Module:
    """
    Apply Post-Training Quantization to the model.
    
    Args:
        model: FP32 model to quantize
        calibration_data: List of input tensors for calibration
        device: Device to run quantization on
    
    Returns:
        Quantized model
    """
    # TODO: Implement PTQ using torch.quantization
    # Steps:
    # 1. Prepare model for quantization (set qconfig)
    # 2. Prepare model (insert observers)
    # 3. Calibrate with calibration_data
    # 4. Convert to quantized model
    
    # Hint: Use quant.quantize_fx for FX Graph Mode quantization
    # Or use quant.prepare and quant.convert for eager mode
    
    pass


def compare_models(
    fp32_model: nn.Module,
    quantized_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """
    Compare FP32 and quantized models on accuracy, speed, and memory.
    
    Args:
        fp32_model: Original FP32 model
        quantized_model: Quantized model
        test_loader: Test data loader
        device: Device to run comparison on
    
    Returns:
        Dictionary with comparison metrics
    """
    # TODO: Implement comprehensive comparison
    # Measure:
    # - Accuracy (both models)
    # - Inference time (both models)
    # - Memory usage (both models)
    # - Speedup ratio
    # - Accuracy drop
    
    # Create dummy input for timing
    dummy_input = next(iter(test_loader))[0][:1].to(device)
    
    results = {}
    
    # TODO: Evaluate accuracy
    # results['fp32_accuracy'] = ...
    # results['quantized_accuracy'] = ...
    # results['accuracy_drop'] = ...
    
    # TODO: Measure inference time
    # results['fp32_time_ms'] = ...
    # results['quantized_time_ms'] = ...
    # results['speedup'] = ...
    
    # TODO: Measure memory
    # results['fp32_memory_mb'] = ...
    # results['quantized_memory_mb'] = ...
    # results['memory_reduction'] = ...
    
    return results



def main():
    """Main function to run the quantization challenge"""
    print("=" * 70)
    print("Challenge 1: Post-Training Quantization")
    print("=" * 70)
    
    # Auto-detect best device: MPS > CUDA > CPU
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create or load a pre-trained model
    # For this exercise, we'll use a simple CNN
    # In practice, you'd load a pre-trained ResNet or BERT
    model = SimpleCNN(num_classes=10).to(device)
    model.eval()
    
    print("\n1. Model created")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data for testing
    # In practice, you'd use real datasets (CIFAR-10, ImageNet, etc.)
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    print("\n2. Measuring FP32 model performance...")
    # TODO: Measure baseline performance
    # fp32_time = measure_inference_time(model, dummy_input)
    # print(f"   FP32 inference time: {fp32_time:.3f} ms")
    
    # TODO: Prepare calibration data
    # For this exercise, create dummy calibration data
    # In practice, use real test data
    print("\n3. Preparing calibration data...")
    # calibration_data = [torch.randn(1, 3, 32, 32).to(device) for _ in range(100)]
    
    # TODO: Apply PTQ
    print("\n4. Applying Post-Training Quantization...")
    # quantized_model = apply_ptq(model, calibration_data, device)
    
    # TODO: Compare models
    print("\n5. Comparing FP32 vs Quantized models...")
    # results = compare_models(model, quantized_model, test_loader, device)
    # print(f"\nResults:")
    # print(f"  Accuracy: FP32={results['fp32_accuracy']:.2f}%, "
    #       f"INT8={results['quantized_accuracy']:.2f}%, "
    #       f"Drop={results['accuracy_drop']:.2f}%")
    # print(f"  Speed: FP32={results['fp32_time_ms']:.3f}ms, "
    #       f"INT8={results['quantized_time_ms']:.3f}ms, "
    #       f"Speedup={results['speedup']:.2f}x")
    # print(f"  Memory: FP32={results['fp32_memory_mb']:.2f}MB, "
    #       f"INT8={results['quantized_memory_mb']:.2f}MB, "
    #       f"Reduction={results['memory_reduction']:.1f}%")
    
    print("\n" + "=" * 70)
    print("TODO: Implement the functions above to complete the challenge")
    print("=" * 70)
    print("\nTips:")
    print("- Read PyTorch quantization docs: https://pytorch.org/docs/stable/quantization.html")
    print("- Use quant.quantize_fx for FX Graph Mode (recommended)")
    print("- Ensure calibration data is representative of real data")
    print("- Measure multiple metrics: accuracy, speed, memory")


if __name__ == "__main__":
    main()

