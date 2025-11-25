import torch
import torch.optim as optim
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SSH/headless environments
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from collections import defaultdict
from PIL import Image
import torch.nn.functional as F

# Dynamic device selection with priority: CUDA -> MPS -> CPU
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

input_dim = 224
input_channels = 3
num_layers = 5  # Increased depth for better capacity
batch_size = 128  # Increased for better GPU utilization
epochs = 30  # More epochs for better convergence
learning_rate = 0.001
weight_decay = 1e-4  # L2 regularization
num_workers = 4  # Parallel data loading
pin_memory = True  # Faster GPU transfer

# Transform - convert grayscale to RGB to handle mixed image formats in Caltech 101
def to_rgb_pil(image):
    """Convert PIL grayscale images to RGB before tensor conversion"""
    if image.mode == 'L':  # Grayscale
        return image.convert('RGB')
    elif image.mode == 'RGB':
        return image
    else:
        return image.convert('RGB')  # Convert any other format to RGB

# Training transforms with data augmentation
train_transform = transforms.Compose([
    transforms.Lambda(to_rgb_pil),  # Convert grayscale to RGB first
    transforms.Resize((input_dim + 32, input_dim + 32)),  # Slightly larger for random crop
    transforms.RandomCrop(input_dim),
    transforms.RandomHorizontalFlip(p=0.5),  # Data augmentation
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Validation/test transforms (no augmentation)
test_transform = transforms.Compose([
    transforms.Lambda(to_rgb_pil),
    transforms.Resize((input_dim, input_dim)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Download Caltech 101 dataset
print("Downloading Caltech 101 dataset...")
caltech101_data = datasets.Caltech101(
    root='./data',
    download=True,
    transform=None  # We'll apply transforms after split
)

dataset_size = len(caltech101_data)
test_size = int(0.1 * dataset_size)
train_size = dataset_size - test_size
num_classes= len(caltech101_data.categories)

print(f"Dataset downloaded successfully!")
print(f"Total number of images: {dataset_size}")
print(f"Total number of training images: {train_size}")
print(f"Total number of testing images: {test_size}")
print(f"Number of classes: {num_classes}")

# Create a custom dataset wrapper to apply different transforms
class TransformDataset:
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        image, label = self.subset[index]
        if self.transform:
            image = self.transform(image)
        return image, label
    
    def __len__(self):
        return len(self.subset)

# Split dataset indices
train_indices, test_indices = random_split(range(len(caltech101_data)), [train_size, test_size])
train_subset = Subset(caltech101_data, train_indices.indices)
test_subset = Subset(caltech101_data, test_indices.indices)

train_dataset = TransformDataset(train_subset, train_transform)
test_dataset = TransformDataset(test_subset, test_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                         num_workers=num_workers, pin_memory=pin_memory)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=pin_memory)

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Increased model capacity for better performance
        # Architecture: 64 -> 128 -> 256 -> 128 -> 64 channels
        self.input = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(64)
        
        self.middle1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn_middle1 = nn.BatchNorm2d(128)
        
        self.middle2 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn_middle2 = nn.BatchNorm2d(256)
        
        self.middle3 = nn.Conv2d(256, 128, 3, padding=1)
        self.bn_middle3 = nn.BatchNorm2d(128)
        
        self.middle4 = nn.Conv2d(128, 64, 3, padding=1)
        self.bn_middle4 = nn.BatchNorm2d(64)
        
        # Global average pooling instead of flattening
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.output = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Input layer
        x = self.relu(self.bn_input(self.input(x)))
        x = nn.MaxPool2d(2)(x)  # 224 -> 112
        
        # Middle layers with increasing then decreasing channels
        x = self.relu(self.bn_middle1(self.middle1(x)))
        x = nn.MaxPool2d(2)(x)  # 112 -> 56
        
        x = self.relu(self.bn_middle2(self.middle2(x)))
        x = nn.MaxPool2d(2)(x)  # 56 -> 28
        
        x = self.relu(self.bn_middle3(self.middle3(x)))
        x = nn.MaxPool2d(2)(x)  # 28 -> 14
        
        x = self.relu(self.bn_middle4(self.middle4(x)))
        x = nn.MaxPool2d(2)(x)  # 14 -> 7
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.output(x)
        return x

model = MyNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# Learning rate scheduler for better convergence
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
criterion = nn.CrossEntropyLoss()

# Print model summary
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel Summary:")
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print(f"Weight decay: {weight_decay}\n")

# Create output directory for saving results
os.makedirs('results', exist_ok=True)

# Track metrics for plotting
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in tqdm(range(epochs), desc="Training Progress", unit="epoch"):
    # Train the model
    model.train()
    correct, total = 0, 0
    train_loss = 0.0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False, unit="batch")
    for images, labels in train_pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(images)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        # Find the accuracy
        predictions = torch.argmax(predictions, dim=1)
        correct += (predictions == labels).sum().item()
        total += len(labels)
        train_loss += loss.item()
        
        # Update progress bar with current metrics
        train_pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{correct / total:.2%}'
        })
    
    train_acc = correct / total
    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)
    print(f"Epoch {epoch+1:2d}/{epochs} | Training   | Loss: {avg_train_loss:.4f} | Accuracy: {train_acc:.2%}")

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        
        test_pbar = tqdm(test_loader, desc=f"Epoch {epoch+1}/{epochs} [Test]", leave=False, unit="batch")
        for images, labels in test_pbar:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images)
            loss = criterion(predictions, labels)
            
            # Find the accuracy
            predictions = torch.argmax(predictions, dim=1)
            correct += (predictions == labels).sum().item()
            total += len(labels)
            test_loss += loss.item()
            
            # Update progress bar with current metrics
            test_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct / total:.2%}'
            })
        
        test_acc = correct / total
        avg_test_loss = test_loss / len(test_loader)
        test_losses.append(avg_test_loss)
        test_accuracies.append(test_acc)
        print(f"Epoch {epoch+1:2d}/{epochs} | Testing    | Loss: {avg_test_loss:.4f} | Accuracy: {test_acc:.2%}")
        
        # Update learning rate scheduler based on validation loss
        scheduler.step(avg_test_loss)

# Save the trained model
model_save_path = 'results/model.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epochs,
    'train_losses': train_losses,
    'train_accuracies': train_accuracies,
    'test_losses': test_losses,
    'test_accuracies': test_accuracies,
    'num_classes': num_classes,
    'categories': caltech101_data.categories
}, model_save_path)
print(f"\nModel saved to {model_save_path}")

# Plot training and testing metrics
def plot_metrics():
    """Plot training and testing metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(train_losses) + 1)
    
    # Plot losses
    ax1.plot(epochs_range, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs_range, test_losses, 'r-', label='Testing Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Testing Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs_range, train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs_range, test_accuracies, 'r-', label='Testing Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Testing Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('results/training_metrics.png', dpi=300, bbox_inches='tight')
    print("Training metrics plot saved to results/training_metrics.png")
    plt.close()

plot_metrics()

# Function to denormalize images for visualization
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize a tensor image"""
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

# Visualize test images with predictions
def visualize_predictions(model, test_loader, device, num_images=16, save_path='results/test_predictions.png'):
    """Visualize test images with their predictions"""
    model.eval()
    
    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, 3, dim=1)
    
    # Select images to visualize (up to num_images)
    num_images = min(num_images, len(images))
    
    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    for idx in range(num_images):
        ax = axes[idx]
        
        # Denormalize and convert to numpy
        img = denormalize(images[idx].cpu()).clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()
        
        # Display image
        ax.imshow(img)
        ax.axis('off')
        
        # Get predictions
        true_label = labels[idx].item()
        pred_label = predicted[idx].item()
        confidence = top_probs[idx][0].item()
        
        # Color: green if correct, red if wrong
        color = 'green' if true_label == pred_label else 'red'
        
        # Title with prediction info
        true_class = caltech101_data.categories[true_label]
        pred_class = caltech101_data.categories[pred_label]
        title = f"True: {true_class}\nPred: {pred_class}\nConf: {confidence:.2%}"
        ax.set_title(title, fontsize=9, color=color, fontweight='bold')
    
    # Hide unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Test Image Predictions (Green=Correct, Red=Incorrect)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Test predictions visualization saved to {save_path}")
    plt.close()

visualize_predictions(model, test_loader, device)

# Analyze per-class performance
def analyze_class_performance(model, test_loader, device, categories):
    """Analyze which classes the model struggles with most"""
    model.eval()
    
    # Track per-class statistics
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_errors = defaultdict(list)  # Store (image, true_label, pred_label, confidence)
    
    all_images = []
    all_labels = []
    all_predictions = []
    all_confidences = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Analyzing class performance"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probabilities = F.softmax(outputs, dim=1)
            confidences = probabilities.max(dim=1)[0]
            
            # Store for later visualization
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(predicted.cpu())
            all_confidences.append(confidences.cpu())
            
            # Track per-class accuracy
            for i in range(len(labels)):
                true_label = labels[i].item()
                pred_label = predicted[i].item()
                confidence = confidences[i].item()
                
                class_total[true_label] += 1
                if true_label == pred_label:
                    class_correct[true_label] += 1
                else:
                    # Store error cases
                    class_errors[true_label].append({
                        'image_idx': len(all_images) - 1,
                        'batch_idx': i,
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'confidence': confidence
                    })
    
    # Calculate per-class accuracy
    class_accuracies = {}
    for class_idx in class_total:
        accuracy = class_correct[class_idx] / class_total[class_idx]
        class_accuracies[class_idx] = accuracy
    
    # Sort classes by accuracy (worst first)
    sorted_classes = sorted(class_accuracies.items(), key=lambda x: x[1])
    
    # Print worst performing classes
    print("\n" + "="*80)
    print("CLASS PERFORMANCE ANALYSIS")
    print("="*80)
    print(f"\nWorst performing classes (bottom 10):")
    print("-" * 80)
    for class_idx, accuracy in sorted_classes[:10]:
        class_name = categories[class_idx]
        total = class_total[class_idx]
        correct = class_correct[class_idx]
        print(f"{class_name:30s} | Accuracy: {accuracy:.2%} | Correct: {correct:3d}/{total:3d}")
    
    print(f"\nBest performing classes (top 10):")
    print("-" * 80)
    for class_idx, accuracy in sorted_classes[-10:][::-1]:
        class_name = categories[class_idx]
        total = class_total[class_idx]
        correct = class_correct[class_idx]
        print(f"{class_name:30s} | Accuracy: {accuracy:.2%} | Correct: {correct:3d}/{total:3d}")
    
    # Visualize worst performing classes with examples
    visualize_worst_classes(all_images, all_labels, all_predictions, all_confidences, 
                           class_errors, sorted_classes[:10], categories, 
                           save_path='results/worst_classes_analysis.png')
    
    return class_accuracies, class_errors

def visualize_worst_classes(all_images, all_labels, all_predictions, all_confidences,
                            class_errors, worst_classes, categories, save_path='results/worst_classes_analysis.png'):
    """Visualize examples from worst performing classes"""
    # Concatenate all batches
    all_images_tensor = torch.cat(all_images, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_predictions_tensor = torch.cat(all_predictions, dim=0)
    all_confidences_tensor = torch.cat(all_confidences, dim=0)
    
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(len(worst_classes), 4, hspace=0.3, wspace=0.2)
    
    for row, (class_idx, accuracy) in enumerate(worst_classes):
        class_name = categories[class_idx]
        
        # Find examples of this class (both correct and incorrect)
        class_mask = (all_labels_tensor == class_idx)
        class_indices = torch.where(class_mask)[0]
        
        # Get some examples (mix of correct and incorrect)
        correct_indices = class_indices[all_predictions_tensor[class_indices] == class_idx]
        incorrect_indices = class_indices[all_predictions_tensor[class_indices] != class_idx]
        
        # Select examples
        num_examples = min(4, len(class_indices))
        selected_indices = []
        
        if len(correct_indices) > 0 and len(incorrect_indices) > 0:
            # Mix of correct and incorrect
            selected_indices = list(correct_indices[:2].cpu().numpy()) + list(incorrect_indices[:2].cpu().numpy())
            selected_indices = selected_indices[:num_examples]
        elif len(correct_indices) > 0:
            selected_indices = correct_indices[:num_examples].cpu().numpy().tolist()
        elif len(incorrect_indices) > 0:
            selected_indices = incorrect_indices[:num_examples].cpu().numpy().tolist()
        
        # Plot examples
        for col in range(4):
            ax = fig.add_subplot(gs[row, col])
            
            if col < len(selected_indices):
                idx = selected_indices[col]
                img = denormalize(all_images_tensor[idx]).clamp(0, 1)
                img = img.permute(1, 2, 0).numpy()
                
                ax.imshow(img)
                ax.axis('off')
                
                true_label = all_labels_tensor[idx].item()
                pred_label = all_predictions_tensor[idx].item()
                confidence = all_confidences_tensor[idx].item()
                
                is_correct = (true_label == pred_label)
                color = 'green' if is_correct else 'red'
                
                true_class = categories[true_label]
                pred_class = categories[pred_label]
                
                if col == 0:
                    # First column: show class name and accuracy
                    title = f"{class_name}\nAccuracy: {accuracy:.2%}"
                    ax.set_title(title, fontsize=10, fontweight='bold', color='black')
                else:
                    title = f"Pred: {pred_class}\nConf: {confidence:.2%}"
                    ax.set_title(title, fontsize=9, color=color)
            else:
                ax.axis('off')
    
    plt.suptitle('Worst Performing Classes - Examples (Green=Correct, Red=Incorrect)', 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nWorst classes analysis saved to {save_path}")
    plt.close()

# Run class performance analysis
class_accuracies, class_errors = analyze_class_performance(model, test_loader, device, caltech101_data.categories)

print("\n" + "="*80)
print("All analysis complete! Results saved in 'results/' directory:")
print("  - model.pth: Saved model")
print("  - training_metrics.png: Training/testing metrics plots")
print("  - test_predictions.png: Sample test predictions")
print("  - worst_classes_analysis.png: Worst performing classes analysis")
print("="*80)


