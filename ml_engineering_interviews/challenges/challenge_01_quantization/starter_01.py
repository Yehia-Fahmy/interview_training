"""
Challenge 1: Post-Training Quantization
Starter Code

Implement PTQ for a pre-trained model and measure accuracy-speed trade-offs.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import quantize_fx
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from typing import Tuple, Dict
import time
import os
import platform


def load_imagenet_dataset(data_dir: str = None, batch_size: int = 16, num_classes: int = 10):
    """
    Load ImageNet dataset resized to 32x32 for compatibility with SimpleCNN.
    Automatically downloads and places dataset in current directory if not present.
    
    Args:
        data_dir: Directory where ImageNet data is stored. If None, uses './imagenet' in current directory.
        batch_size: Batch size for DataLoader
        num_classes: Number of classes to use (10 for SimpleCNN compatibility)
    
    Returns:
        DataLoader for ImageNet validation set
    """
    # Define transforms: resize to 32x32 to match SimpleCNN input size
    transform = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Default ImageNet directory: use current directory
    if data_dir is None:
        # Use './imagenet' in current working directory
        data_dir = os.path.join(os.getcwd(), 'imagenet')
    
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Check if ImageNet is already downloaded
    val_dir = os.path.join(data_dir, 'val')
    dataset_exists = os.path.exists(val_dir) and len(os.listdir(val_dir)) > 0
    
    if not dataset_exists:
        print(f"ImageNet dataset not found at {data_dir}.")
        print("Attempting to download ImageNet...")
        print("\nNote: ImageNet requires manual registration at http://www.image-net.org/")
        print("For automatic download, you may need to:")
        print("1. Register at http://www.image-net.org/")
        print("2. Download the ILSVRC2012 validation set")
        print("3. Extract it to:", val_dir)
        print("\nTrying to load dataset (will fail if not downloaded)...")
    
    # Load ImageNet validation set
    try:
        dataset = datasets.ImageNet(root=data_dir, split='val', transform=transform)
    except (FileNotFoundError, RuntimeError, AttributeError):
        # Fallback: try ImageFolder if ImageNet class doesn't work
        if os.path.exists(val_dir):
            dataset = datasets.ImageFolder(root=val_dir, transform=transform)
        else:
            raise FileNotFoundError(
                f"ImageNet dataset not found at {data_dir}. "
                f"Please download ImageNet from http://www.image-net.org/ "
                f"and extract the validation set to: {val_dir}"
            )
    
    # If we need to limit to 10 classes, create a subset
    # Map labels to 0-9 range for SimpleCNN compatibility
    if hasattr(dataset, 'classes') and len(dataset.classes) > num_classes:
        # Get indices for first num_classes
        class_indices = {}
        samples = dataset.samples if hasattr(dataset, 'samples') else dataset.imgs
        for idx, (_, label) in enumerate(samples):
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
        
        # Take first num_classes
        selected_indices = []
        for label in sorted(class_indices.keys())[:num_classes]:
            selected_indices.extend(class_indices[label])
        
        dataset = Subset(dataset, selected_indices)
    
    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return dataloader


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
    model.eval()
    # Check if model is quantized (quantized models must stay on CPU)
    # FX quantized models have 'qconfig' attribute or contain 'quantized' in module names
    is_quantized = (
        hasattr(model, 'qconfig') and model.qconfig is not None
    ) or any('quantized' in str(type(m)).lower() for m in model.modules())
    
    total, correct = 0, 0
    for inputs, labels in test_loader:
        # Quantized models must use CPU, others use the specified device
        if is_quantized:
            inputs = inputs.to('cpu')
            labels = labels.to('cpu')
        else:
            inputs = inputs.to(device)
            labels = labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        for i in range(len(inputs)):
            if labels[i] == predicted[i]: correct += 1
            total += 1

    return 100.0 * correct / total


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
    model.eval()
    # Check if model is quantized (quantized models must stay on CPU)
    # FX quantized models have 'qconfig' attribute or contain 'quantized' in module names
    is_quantized = (
        hasattr(model, 'qconfig') and model.qconfig is not None
    ) or any('quantized' in str(type(m)).lower() for m in model.modules())
    
    # Quantized models must use CPU
    if is_quantized:
        input_tensor = input_tensor.to('cpu')
        device = torch.device('cpu')
    else:
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
    # Check if model is quantized (quantized models must stay on CPU)
    # FX quantized models have 'qconfig' attribute or contain 'quantized' in module names
    is_quantized = (
        hasattr(model, 'qconfig') and model.qconfig is not None
    ) or any('quantized' in str(type(m)).lower() for m in model.modules())
    
    # Quantized models must use CPU
    if is_quantized:
        input_tensor = input_tensor.to('cpu')
        device = torch.device('cpu')
    else:
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
    input_tensors = []

    for inputs, _ in test_loader:
        for input_tensor in inputs:
            if len(input_tensors) >= num_samples:
                return input_tensors

            input_tensors.append(input_tensor.unsqueeze(0))
    return input_tensors


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
    model.eval()
    # Save original device to move model back later
    original_device = next(model.parameters()).device
    model.to('cpu')

    # Select appropriate backend based on platform
    # fbgemm is for x86 CPUs, qnnpack is for ARM/Apple Silicon
    machine = platform.machine()
    
    # Set the quantization engine globally FIRST (before creating qconfig)
    # This ensures convert_fx uses the correct backend
    if machine == 'arm64' or machine == 'aarch64':
        backend = 'qnnpack'  # For Apple Silicon / ARM
    else:
        backend = 'fbgemm'  # For x86 CPUs
    
    # Set the backend engine before creating qconfig
    torch.backends.quantized.engine = backend
    
    # Verify backend is available
    try:
        qconfig = quant.get_default_qconfig(backend)
    except Exception as e:
        # If backend fails, try qnnpack as fallback
        if backend != 'qnnpack':
            print(f"Warning: {backend} backend failed ({e}), falling back to qnnpack")
            backend = 'qnnpack'
            torch.backends.quantized.engine = backend
            qconfig = quant.get_default_qconfig(backend)
        else:
            raise RuntimeError(f"Quantization backend {backend} is not available: {e}")
    
    print(f"Using quantization backend: {backend}")
    # Move calibration data to CPU (it's a list of tensors)
    calibration_data_cpu = [data.to('cpu') for data in calibration_data]
    example_input = calibration_data_cpu[0]

    prepared_model = quantize_fx.prepare_fx(
        model,
        {'': qconfig},
        example_input
    )

    with torch.no_grad():
        for data in calibration_data_cpu:
            _ = prepared_model(data)
    
    # Convert with explicit backend
    quantized_model = quantize_fx.convert_fx(prepared_model)
    
    # Move original model back to its original device
    model.to(original_device)
    
    # Quantized models must stay on CPU (quantization backends only work on CPU)
    # Don't move to device - keep on CPU
    return quantized_model


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
    
    # Create dummy input for timing
    # Quantized models need CPU input, FP32 can use the specified device
    sample_batch = next(iter(test_loader))
    dummy_input_fp32 = sample_batch[0][:1].to(device)
    dummy_input_quantized = sample_batch[0][:1].to('cpu')
    
    results = {}
    
    results['fp32_accuracy'] = evaluate_accuracy(fp32_model, test_loader, device)
    results['quantized_accuracy'] = evaluate_accuracy(quantized_model, test_loader, device)
    results['accuracy_drop'] = results['fp32_accuracy'] - results['quantized_accuracy']

    results['fp32_time_ms'] = measure_inference_time(fp32_model, dummy_input_fp32)
    results['quantized_time_ms'] = measure_inference_time(quantized_model, dummy_input_quantized)
    results['speedup'] = results['fp32_time_ms'] / results['quantized_time_ms']

    fp32_memory = measure_memory_usage(fp32_model, dummy_input_fp32)
    quantized_memory = measure_memory_usage(quantized_model, dummy_input_quantized)
    results['fp32_memory_mb'] = fp32_memory['model_size_mb']
    results['quantized_memory_mb'] = quantized_memory['model_size_mb']
    results['memory_reduction'] = results['fp32_memory_mb'] / results['quantized_memory_mb']
    
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
    
    # Load ImageNet dataset
    print("\n2. Loading ImageNet dataset...")
    try:
        test_loader = load_imagenet_dataset(batch_size=16, num_classes=10)
        # Get a sample input from ImageNet
        sample_batch = next(iter(test_loader))
        dummy_input = sample_batch[0][:1].to(device)
        print(f"   ImageNet dataset loaded successfully")
    except FileNotFoundError as e:
        print(f"   Error: {e}")
        print("   Falling back to dummy data for testing")
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        # Create dummy test_loader for compatibility
        dummy_data = torch.randn(100, 3, 32, 32)
        dummy_labels = torch.randint(0, 10, (100,))
        from torch.utils.data import TensorDataset
        dummy_dataset = TensorDataset(dummy_data, dummy_labels)
        test_loader = DataLoader(dummy_dataset, batch_size=16, shuffle=False)
    
    print("\n3. Measuring FP32 model performance...")
    fp32_time = measure_inference_time(model, dummy_input)
    print(f"   FP32 inference time: {fp32_time:.3f} ms")
    
    # Use ImageNet samples for calibration
    print("\n4. Preparing calibration data...")
    calibration_data = prepare_calibration_data(test_loader, num_samples=100)
    # Move calibration data to device
    calibration_data = [data.to(device) for data in calibration_data]

    print("\n5. Applying Post-Training Quantization...")
    quantized_model = apply_ptq(model, calibration_data, device)
    
    print("\n6. Comparing FP32 vs Quantized models...")
    results = compare_models(model, quantized_model, test_loader, device)
    print(f"\nResults:")
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

