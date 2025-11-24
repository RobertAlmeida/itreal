import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# ===== Transformações =====
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# ===== Dataset =====
train_dataset = datasets.ImageFolder("./dataset", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# ===== Modelo =====
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===== Loss e Otimizador =====
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0001)

# ===== Treino =====
EPOCHS = 10
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader)}")

# ===== Salvar modelo =====
torch.save(model.state_dict(), "ai_detector_model.pth")

print("Treinamento finalizado e modelo salvo!")
