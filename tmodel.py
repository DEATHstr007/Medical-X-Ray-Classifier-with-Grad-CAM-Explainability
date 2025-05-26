import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

# Initialize device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model
class XRayClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.base = torch.hub.load('pytorch/vision', 'densenet121', pretrained=True)
        for param in self.base.parameters():
            param.requires_grad = False
        num_features = self.base.classifier.in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base(x)

# Initialize model and move to device
model = XRayClassifier().to(device)

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(root='chest_xray/train', transform=transform)
val_dataset = datasets.ImageFolder(root='chest_xray/val', transform=transform)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Training loop
best_val_acc = 0
for epoch in range(15):
    model.train()
    train_loss = 0
    
    # Training phase
    for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_acc = correct / total
    print(f'\nTrain Loss: {train_loss/len(train_loader):.4f}, Val Acc: {val_acc:.4f}')
    
    # Save best model
    if val_acc > best_val_acc:
        torch.save(model.state_dict(), 'best_model.pth')
        best_val_acc = val_acc
    
    scheduler.step()

print(f'\nTraining complete. Best validation accuracy: {best_val_acc:.4f}')
