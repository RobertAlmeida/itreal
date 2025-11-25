import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR

# ===== DATA AUGMENTATION =====
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ===== DATASET ÚNICO =====
dataset = datasets.ImageFolder("./dataset", transform=train_transform)

# ===== SPLIT =====
val_ratio = 0.2
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ===== MODEL =====
model = models.efficientnet_b0(pretrained=True)

# Fine-tuning completo: DESCONGELAR todas as camadas
for param in model.parameters():
    param.requires_grad = True

# Saída final com 2 classes (ia / real)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===== LOSS, OPTIMIZER E SCHEDULER =====
criterion = nn.CrossEntropyLoss()

# LR muito baixo para fine-tuning completo
optimizer = Adam(model.parameters(), lr=1e-5)

# Scheduler suave para fine-tuning
scheduler = CosineAnnealingLR(optimizer, T_max=10)

# ===== TRAIN =====
EPOCHS = 150
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_correct = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(imgs)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, preds = torch.max(output, 1)
        train_correct += torch.sum(preds == labels).item()

    train_acc = train_correct / train_size

    # ===== VALIDAÇÃO =====
    model.eval()
    val_loss = 0
    val_correct = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            output = model(imgs)
            loss = criterion(output, labels)

            val_loss += loss.item()
            _, preds = torch.max(output, 1)
            val_correct += torch.sum(preds == labels).item()

    val_acc = val_correct / val_size
    scheduler.step()

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss/len(train_loader):.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss/len(val_loader):.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )

# ===== SAVE =====
torch.save(model.state_dict(), "app/models/ai_detector_model.pth")
print("Fine-tuning completo concluído e modelo salvo!")
