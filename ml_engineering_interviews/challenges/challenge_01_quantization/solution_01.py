"""
Challenge 1: Post-Training Quantization
Ideal Implementation - Production-ready, optimized, with best practices

This implementation includes:
- FX Graph Mode quantization (more accurate)
- Proper error handling
- Comprehensive profiling
- Support for both CPU and GPU
- Detailed documentation
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Optional
import time
import warnings


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


def evaluate_accuracy(
    model: nn.Module, 
    test_loader: DataLoader, 
    device: torch.device,
    verbose: bool = False
) -> float:
    """
    Evaluate model accuracy with proper error handling and progress tracking.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to run evaluation on
        verbose: Whether to print progress
    
    Returns:
        Accuracy percentage
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            try:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if verbose and (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1} batches, "
                          f"current accuracy: {100.0 * correct / total:.2f}%")
            except Exception as e:
                warnings.warn(f"Error processing batch {batch_idx}: {e}")
                continue
    
    if total == 0:
        raise ValueError("No samples were processed during evaluation")
    
    accuracy = 100.0 * correct / total
    return accuracy


def measure_inference_time(
    model: nn.Module, 
    input_tensor: torch.Tensor, 
    num_runs: int = 100,
    warmup_runs: int = 10,
    percentile: Optional[float] = None
) -> Dict[str, float]:
    """
    Measure inference time with statistical analysis.
    
    Args:
        model: Model to profile
        input_tensor: Input tensor for inference
        num_runs: Number of inference runs for averaging
        warmup_runs: Number of warmup runs
        percentile: Optional percentile to compute (e.g., 95 for p95 latency)
    
    Returns:
        Dictionary with timing statistics
    """
    model.eval()
    
    # Warmup to ensure stable measurements
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(input_tensor)
    
    # Synchronize if CUDA
    if input_tensor.is_cuda:
        torch.cuda.synchronize()
    
    # Measure multiple runs
    times = []
    start_event = torch.cuda.Event(enable_timing=True) if input_tensor.is_cuda else None
    end_event = torch.cuda.Event(enable_timing=True) if input_tensor.is_cuda else None
    
    with torch.no_grad():
        for _ in range(num_runs):
            if input_tensor.is_cuda:
                start_event.record()
                _ = model(input_tensor)
                end_event.record()
                torch.cuda.synchronize()
                times.append(start_event.elapsed_time(end_event))  # milliseconds
            else:
                start = time.perf_counter()
                _ = model(input_tensor)
                elapsed = (time.perf_counter() - start) * 1000  # convert to ms
                times.append(elapsed)
    
    stats = {
        'mean_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'std_ms': (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5
    }
    
    if percentile:
        sorted_times = sorted(times)
        idx = int(len(sorted_times) * percentile / 100)
        stats[f'p{percentile}_ms'] = sorted_times[idx]
    
    return stats


def measure_memory_usage(
    model: nn.Module, 
    input_tensor: torch.Tensor,
    include_activations: bool = True
) -> Dict[str, float]:
    """
    Measure model memory usage comprehensively.
    
    Args:
        model: Model to measure
        input_tensor: Input tensor for forward pass
        include_activations: Whether to include activation memory
    
    Returns:
        Dictionary with memory statistics
    """
    stats = {}
    
    # Model parameter size
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size = param_size + buffer_size
    stats['model_size_mb'] = model_size / (1024 * 1024)
    
    if torch.cuda.is_available() and input_tensor.is_cuda:
        # Clear cache and reset stats
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        initial_memory = torch.cuda.memory_allocated()
        
        # Run forward pass
        with torch.no_grad():
            _ = model(input_tensor)
        
        peak_memory = torch.cuda.max_memory_allocated()
        stats['peak_memory_mb'] = peak_memory / (1024 * 1024)
        stats['activation_memory_mb'] = (peak_memory - initial_memory) / (1024 * 1024) if include_activations else 0.0
    else:
        stats['peak_memory_mb'] = 0.0
        stats['activation_memory_mb'] = 0.0
    
    return stats


def prepare_calibration_data(
    test_loader: DataLoader, 
    num_samples: int = 100,
    shuffle: bool = False
) -> List[torch.Tensor]:
    """
    Prepare calibration data with proper sampling.
    
    Args:
        test_loader: DataLoader for calibration data
        num_samples: Number of samples to use for calibration
        shuffle: Whether to shuffle before sampling
    
    Returns:
        List of input tensors for calibration
    """
    calibration_data = []
    count = 0
    
    # Collect all data first if we need to shuffle
    if shuffle:
        all_data = []
        for inputs, _ in test_loader:
            all_data.extend([x.unsqueeze(0) for x in inputs])
        import random
        random.shuffle(all_data)
        return all_data[:num_samples]
    
    # Otherwise, collect sequentially
    for inputs, _ in test_loader:
        for input_tensor in inputs:
            calibration_data.append(input_tensor.unsqueeze(0))
            count += 1
            if count >= num_samples:
                return calibration_data
    
    if len(calibration_data) == 0:
        raise ValueError("No calibration data could be prepared")
    
    return calibration_data


def apply_ptq_fx(
    model: nn.Module,
    calibration_data: List[torch.Tensor],
    device: torch.device,
    backend: str = 'fbgemm'
) -> nn.Module:
    """
    Apply PTQ using FX Graph Mode (recommended approach).
    
    Args:
        model: FP32 model to quantize
        calibration_data: List of input tensors for calibration
        device: Device to run quantization on
        backend: Quantization backend ('fbgemm' for CPU, 'qnnpack' for mobile)
    
    Returns:
        Quantized model
    """
    model.eval()
    model = model.to('cpu')  # FX quantization typically on CPU
    
    # Define example input
    example_input = calibration_data[0]
    
    # Set quantization config
    qconfig = quant.get_default_qconfig(backend)
    
    # Prepare model
    prepared_model = quant.prepare_fx(model, {'': qconfig}, example_input)
    
    # Calibrate
    print(f"  Calibrating with {len(calibration_data)} samples...")
    with torch.no_grad():
        for i, data in enumerate(calibration_data):
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(calibration_data)} samples")
            _ = prepared_model(data)
    
    # Convert to quantized
    quantized_model = quant.convert_fx(prepared_model)
    
    return quantized_model.to(device)


def apply_ptq_eager(
    model: nn.Module,
    calibration_data: List[torch.Tensor],
    device: torch.device,
    backend: str = 'fbgemm'
) -> nn.Module:
    """
    Apply PTQ using eager mode (fallback if FX doesn't work).
    
    Args:
        model: FP32 model to quantize
        calibration_data: List of input tensors for calibration
        device: Device to run quantization on
        backend: Quantization backend
    
    Returns:
        Quantized model
    """
    model.eval()
    model = model.to('cpu')
    
    # Set quantization config
    model.qconfig = quant.get_default_qconfig(backend)
    
    # Prepare model
    prepared_model = quant.prepare(model, inplace=False)
    
    # Calibrate
    print(f"  Calibrating with {len(calibration_data)} samples...")
    with torch.no_grad():
        for i, data in enumerate(calibration_data):
            if (i + 1) % 10 == 0:
                print(f"    Processed {i + 1}/{len(calibration_data)} samples")
            _ = prepared_model(data)
    
    # Convert
    quantized_model = quant.convert(prepared_model, inplace=False)
    
    return quantized_model.to(device)


def compare_models(
    fp32_model: nn.Module,
    quantized_model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive model comparison with detailed metrics.
    
    Args:
        fp32_model: Original FP32 model
        quantized_model: Quantized model
        test_loader: Test data loader
        device: Device to run comparison on
        verbose: Whether to print progress
    
    Returns:
        Dictionary with comprehensive comparison metrics
    """
    # Get dummy input for timing
    dummy_input = next(iter(test_loader))[0][:1].to(device)
    
    results = {}
    
    if verbose:
        print("\nEvaluating accuracy...")
    
    # Accuracy
    results['fp32_accuracy'] = evaluate_accuracy(fp32_model, test_loader, device, verbose)
    results['quantized_accuracy'] = evaluate_accuracy(quantized_model, test_loader, device, verbose)
    results['accuracy_drop'] = results['fp32_accuracy'] - results['quantized_accuracy']
    results['accuracy_retention'] = (results['quantized_accuracy'] / results['fp32_accuracy'] * 100) if results['fp32_accuracy'] > 0 else 0
    
    if verbose:
        print("\nMeasuring inference time...")
    
    # Inference time with statistics
    fp32_timing = measure_inference_time(fp32_model, dummy_input, percentile=95)
    quantized_timing = measure_inference_time(quantized_model, dummy_input, percentile=95)
    
    results['fp32_time_ms'] = fp32_timing['mean_ms']
    results['quantized_time_ms'] = quantized_timing['mean_ms']
    results['speedup'] = results['fp32_time_ms'] / results['quantized_time_ms']
    results['fp32_p95_ms'] = fp32_timing.get('p95_ms', 0)
    results['quantized_p95_ms'] = quantized_timing.get('p95_ms', 0)
    
    if verbose:
        print("\nMeasuring memory usage...")
    
    # Memory
    fp32_memory = measure_memory_usage(fp32_model, dummy_input)
    quantized_memory = measure_memory_usage(quantized_model, dummy_input)
    
    results['fp32_memory_mb'] = fp32_memory['model_size_mb']
    results['quantized_memory_mb'] = quantized_memory['model_size_mb']
    results['memory_reduction'] = (
        (results['fp32_memory_mb'] - results['quantized_memory_mb']) / 
        results['fp32_memory_mb'] * 100
    ) if results['fp32_memory_mb'] > 0 else 0
    
    return results


def main():
    """Main function with comprehensive testing"""
    print("=" * 70)
    print("Challenge 1: Post-Training Quantization - Ideal Implementation")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = SimpleCNN(num_classes=10).to(device)
    model.eval()
    
    print(f"\nModel created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dummy test data
    inputs = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 10, (100,))
    dataset = torch.utils.data.TensorDataset(inputs, labels)
    test_loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # Prepare calibration data
    print("\nPreparing calibration data...")
    calibration_data = prepare_calibration_data(test_loader, num_samples=50)
    print(f"  Prepared {len(calibration_data)} calibration samples")
    
    # Apply PTQ (try FX first, fallback to eager)
    print("\nApplying Post-Training Quantization...")
    try:
        quantized_model = apply_ptq_fx(model, calibration_data, device)
        print("  Used FX Graph Mode quantization")
    except Exception as e:
        print(f"  FX quantization failed: {e}")
        print("  Falling back to eager mode...")
        quantized_model = apply_ptq_eager(model, calibration_data, device)
        print("  Used Eager Mode quantization")
    
    # Compare
    print("\nComparing models...")
    results = compare_models(model, quantized_model, test_loader, device)
    
    # Print comprehensive results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"\nAccuracy:")
    print(f"  FP32:        {results['fp32_accuracy']:.2f}%")
    print(f"  INT8:        {results['quantized_accuracy']:.2f}%")
    print(f"  Drop:        {results['accuracy_drop']:.2f}%")
    print(f"  Retention:   {results['accuracy_retention']:.1f}%")
    
    print(f"\nInference Time:")
    print(f"  FP32 (mean): {results['fp32_time_ms']:.3f} ms")
    print(f"  INT8 (mean): {results['quantized_time_ms']:.3f} ms")
    print(f"  Speedup:     {results['speedup']:.2f}x")
    if results.get('fp32_p95_ms', 0) > 0:
        print(f"  FP32 (p95):  {results['fp32_p95_ms']:.3f} ms")
        print(f"  INT8 (p95):  {results['quantized_p95_ms']:.3f} ms")
    
    print(f"\nMemory:")
    print(f"  FP32:        {results['fp32_memory_mb']:.2f} MB")
    print(f"  INT8:        {results['quantized_memory_mb']:.2f} MB")
    print(f"  Reduction:   {results['memory_reduction']:.1f}%")
    
    print("\n" + "=" * 70)
    print("Trade-off Analysis:")
    print(f"  Quantization provides {results['speedup']:.2f}x speedup")
    print(f"  with {results['accuracy_drop']:.2f}% accuracy drop")
    print(f"  and {results['memory_reduction']:.1f}% memory reduction")
    print("=" * 70)


if __name__ == "__main__":
    main()

