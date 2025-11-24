import torch
import torch.optim as optim
import torchvision
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for SSH/headless environments
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
num_layers = 3
batch_size = 32
epochs = 15

# Transform - convert grayscale to RGB to handle mixed image formats in Caltech 101
def to_rgb_pil(image):
    """Convert PIL grayscale images to RGB before tensor conversion"""
    if image.mode == 'L':  # Grayscale
        return image.convert('RGB')
    elif image.mode == 'RGB':
        return image
    else:
        return image.convert('RGB')  # Convert any other format to RGB

transform = transforms.Compose([
    transforms.Lambda(to_rgb_pil),  # Convert grayscale to RGB first
    transforms.Resize((input_dim, input_dim)),  # Resize images to a standard size
    transforms.ToTensor(),
])

# Download Caltech 101 dataset
print("Downloading Caltech 101 dataset...")
caltech101_data = datasets.Caltech101(
    root='./data',
    download=True,
    transform=transform
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

train_dataset, test_dataset = random_split(caltech101_data, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size, True)
test_loader = DataLoader(test_dataset, batch_size, True)

class MyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # (input_size - kernel_size + 2*padding) / stride + 1
        self.input = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.middle1 = nn.Conv2d(16, 32, 3, padding=1)
        self.middle = nn.Conv2d(32, 32, 3, padding=1)
        self.last = nn.Conv2d(32, 16, 3, padding=1)
        self.output = nn.Linear(input_dim * input_dim * 16, num_classes)
        self.Relu = nn.ReLU()
    
    def forward(self, x):
        x = self.Relu(self.input(x))

        for i in range(num_layers):
            if i == 0:
                x = self.Relu(self.middle1(x))
            else:
                x = self.Relu(self.middle(x))
        
        x = self.Relu(self.last(x))
        x = x.view(x.size(0), -1)
        x = self.output(x)
        return x

model = MyNetwork().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

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
        print(f"Epoch {epoch+1:2d}/{epochs} | Testing    | Loss: {avg_test_loss:.4f} | Accuracy: {test_acc:.2%}")


