"""
Test suite for Challenge 1: Post-Training Quantization
"""

import torch
import torch.nn as nn
from starter_01 import (
    SimpleCNN,
    evaluate_accuracy,
    measure_inference_time,
    measure_memory_usage,
    prepare_calibration_data,
    apply_ptq,
    compare_models,
)
from torch.utils.data import DataLoader, TensorDataset
import pytest


@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def model(device):
    model = SimpleCNN(num_classes=10).to(device)
    model.eval()
    return model


@pytest.fixture
def test_data(device):
    """Create dummy test dataset"""
    inputs = torch.randn(100, 3, 32, 32)
    labels = torch.randint(0, 10, (100,))
    dataset = TensorDataset(inputs, labels)
    return DataLoader(dataset, batch_size=16, shuffle=False)


def test_model_forward(model, device):
    """Test that model can perform forward pass"""
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    output = model(dummy_input)
    assert output.shape == (1, 10), f"Expected output shape (1, 10), got {output.shape}"


def test_evaluate_accuracy(model, test_data, device):
    """Test accuracy evaluation function"""
    accuracy = evaluate_accuracy(model, test_data, device)
    assert 0 <= accuracy <= 100, f"Accuracy should be between 0 and 100, got {accuracy}"
    assert isinstance(accuracy, float), "Accuracy should be a float"


def test_measure_inference_time(model, device):
    """Test inference time measurement"""
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    time_ms = measure_inference_time(model, dummy_input, num_runs=10)
    assert time_ms > 0, "Inference time should be positive"
    assert isinstance(time_ms, float), "Inference time should be a float"


def test_measure_memory_usage(model, device):
    """Test memory usage measurement"""
    if not torch.cuda.is_available():
        pytest.skip("Memory measurement requires CUDA")
    
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    memory_stats = measure_memory_usage(model, dummy_input)
    
    assert 'model_size_mb' in memory_stats, "Should include model_size_mb"
    assert 'peak_memory_mb' in memory_stats, "Should include peak_memory_mb"
    assert memory_stats['model_size_mb'] > 0, "Model size should be positive"
    assert memory_stats['peak_memory_mb'] > 0, "Peak memory should be positive"


def test_prepare_calibration_data(test_data):
    """Test calibration data preparation"""
    calibration_data = prepare_calibration_data(test_data, num_samples=50)
    assert len(calibration_data) <= 50, "Should not exceed num_samples"
    assert all(isinstance(x, torch.Tensor) for x in calibration_data), "All items should be tensors"
    assert all(x.dim() == 4 for x in calibration_data), "All tensors should be 4D (B, C, H, W)"


def test_apply_ptq(model, test_data, device):
    """Test PTQ application"""
    calibration_data = prepare_calibration_data(test_data, num_samples=20)
    quantized_model = apply_ptq(model, calibration_data, device)
    
    assert quantized_model is not None, "Quantized model should not be None"
    
    # Test that quantized model can perform forward pass
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    output = quantized_model(dummy_input)
    assert output.shape == (1, 10), f"Quantized model output shape should be (1, 10), got {output.shape}"


def test_compare_models(model, test_data, device):
    """Test model comparison function"""
    # Create a quantized model for comparison
    calibration_data = prepare_calibration_data(test_data, num_samples=20)
    quantized_model = apply_ptq(model, calibration_data, device)
    
    results = compare_models(model, quantized_model, test_data, device)
    
    # Check required metrics
    required_metrics = [
        'fp32_accuracy', 'quantized_accuracy', 'accuracy_drop',
        'fp32_time_ms', 'quantized_time_ms', 'speedup',
        'fp32_memory_mb', 'quantized_memory_mb', 'memory_reduction'
    ]
    
    for metric in required_metrics:
        assert metric in results, f"Results should include {metric}"
        assert isinstance(results[metric], (int, float)), f"{metric} should be numeric"
    
    # Check logical constraints
    assert 0 <= results['fp32_accuracy'] <= 100, "FP32 accuracy should be 0-100"
    assert 0 <= results['quantized_accuracy'] <= 100, "Quantized accuracy should be 0-100"
    assert results['fp32_time_ms'] > 0, "FP32 time should be positive"
    assert results['quantized_time_ms'] > 0, "Quantized time should be positive"
    assert results['speedup'] > 0, "Speedup should be positive"
    assert results['memory_reduction'] >= 0, "Memory reduction should be non-negative"


def test_quantization_improves_speed(model, test_data, device):
    """Test that quantization improves inference speed (on CPU, INT8 should be faster)"""
    calibration_data = prepare_calibration_data(test_data, num_samples=20)
    quantized_model = apply_ptq(model, calibration_data, device)
    
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    fp32_time = measure_inference_time(model, dummy_input, num_runs=50)
    quantized_time = measure_inference_time(quantized_model, dummy_input, num_runs=50)
    
    # On CPU, INT8 should generally be faster (or at least not much slower)
    # On GPU, this depends on hardware support
    if device.type == 'cpu':
        # Allow some tolerance for measurement variance
        assert quantized_time <= fp32_time * 1.2, \
            f"Quantized model should be faster on CPU. FP32: {fp32_time:.3f}ms, INT8: {quantized_time:.3f}ms"


def test_quantization_reduces_memory(model, test_data, device):
    """Test that quantization reduces model memory footprint"""
    if not torch.cuda.is_available():
        pytest.skip("Memory test requires CUDA")
    
    calibration_data = prepare_calibration_data(test_data, num_samples=20)
    quantized_model = apply_ptq(model, calibration_data, device)
    
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    
    fp32_memory = measure_memory_usage(model, dummy_input)
    quantized_memory = measure_memory_usage(quantized_model, dummy_input)
    
    # Quantized model should use less memory (or at least not more)
    assert quantized_memory['model_size_mb'] <= fp32_memory['model_size_mb'] * 1.1, \
        "Quantized model should use less memory"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

