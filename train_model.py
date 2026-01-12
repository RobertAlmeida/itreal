"""
Script de treinamento do modelo de detecção de IA com recursos avançados.
Inclui early stopping, checkpointing, mixed precision training e logging detalhado.
"""
import torch
import torch.nn as nn
from torch.optim import Adam
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from pathlib import Path
import json


# ===== CONFIGURAÇÕES =====
class TrainingConfig:
    """Configurações de treinamento"""
    # Dados
    dataset_path = "./dataset"
    val_ratio = 0.2
    batch_size = 32
    num_workers = 4
    
    # Modelo
    model_name = "efficientnet_b0"
    num_classes = 2
    pretrained = True
    
    # Treinamento
    epochs = 150
    learning_rate = 1e-5
    weight_decay = 1e-4
    
    # Early Stopping
    early_stopping_patience = 15
    min_delta = 0.001
    
    # Checkpointing
    checkpoint_dir = "checkpoints"
    save_best_only = True
    
    # Mixed Precision
    use_amp = True
    
    # Logging
    log_dir = "runs"
    log_interval = 10
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"


config = TrainingConfig()


# ===== CRIAR DIRETÓRIOS =====
Path(config.checkpoint_dir).mkdir(exist_ok=True)
Path(config.log_dir).mkdir(exist_ok=True)
Path("app/models").mkdir(parents=True, exist_ok=True)


# ===== TENSORBOARD =====
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
writer = SummaryWriter(f"{config.log_dir}/experiment_{timestamp}")


# ===== DATA AUGMENTATION =====
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(256, scale=(0.7, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ===== DATASET =====
print("Loading dataset...")
dataset = datasets.ImageFolder(config.dataset_path, transform=train_transform)
print(f"Total images: {len(dataset)}")
print(f"Classes: {dataset.classes}")

# Split dataset
val_size = int(len(dataset) * config.val_ratio)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_dataset.dataset.transform = train_transform
val_dataset.dataset.transform = val_transform

print(f"Training images: {train_size}")
print(f"Validation images: {val_size}")

# DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=config.batch_size,
    shuffle=False,
    num_workers=config.num_workers,
    pin_memory=True
)


# ===== MODEL =====
print(f"\nInitializing {config.model_name}...")
model = models.efficientnet_b0(pretrained=config.pretrained)

# Descongelar todas as camadas para fine-tuning completo
for param in model.parameters():
    param.requires_grad = True

# Modificar camada de classificação
model.classifier[1] = nn.Linear(model.classifier[1].in_features, config.num_classes)

model = model.to(config.device)
print(f"Model loaded on {config.device}")


# ===== LOSS, OPTIMIZER, SCHEDULER =====
criterion = nn.CrossEntropyLoss()
optimizer = Adam(
    model.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay
)

# Scheduler que reduz LR quando val_loss para de melhorar
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

# Mixed Precision Scaler
scaler = GradScaler() if config.use_amp else None


# ===== EARLY STOPPING =====
class EarlyStopping:
    """Early stopping para evitar overfitting"""
    
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


early_stopping = EarlyStopping(
    patience=config.early_stopping_patience,
    min_delta=config.min_delta
)


# ===== CHECKPOINT MANAGER =====
class CheckpointManager:
    """Gerencia salvamento de checkpoints"""
    
    def __init__(self, checkpoint_dir, save_best_only=True):
        self.checkpoint_dir = checkpoint_dir
        self.save_best_only = save_best_only
        self.best_val_loss = float('inf')
        
    def save_checkpoint(self, epoch, model, optimizer, val_loss, val_acc, metrics):
        """Salva checkpoint do modelo"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'metrics': metrics
        }
        
        # Salvar último checkpoint
        last_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pth')
        torch.save(checkpoint, last_path)
        
        # Salvar melhor checkpoint
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            best_path = os.path.join(self.checkpoint_dir, 'best_checkpoint.pth')
            torch.save(checkpoint, best_path)
            print(f"✓ Best checkpoint saved! Val Loss: {val_loss:.4f}")
            return True
        
        return False


checkpoint_manager = CheckpointManager(
    config.checkpoint_dir,
    save_best_only=config.save_best_only
)


# ===== MÉTRICAS =====
def calculate_metrics(outputs, labels):
    """Calcula métricas de classificação"""
    _, preds = torch.max(outputs, 1)
    correct = torch.sum(preds == labels).item()
    total = labels.size(0)
    accuracy = correct / total
    
    return {
        'correct': correct,
        'total': total,
        'accuracy': accuracy
    }


# ===== TRAINING LOOP =====
print("\n" + "="*50)
print("STARTING TRAINING")
print("="*50 + "\n")

best_val_acc = 0.0
training_history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'learning_rates': []
}

for epoch in range(config.epochs):
    print(f"\nEpoch {epoch+1}/{config.epochs}")
    print("-" * 50)
    
    # ===== TRAINING PHASE =====
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs, labels = imgs.to(config.device), labels.to(config.device)
        
        optimizer.zero_grad()
        
        # Mixed Precision Training
        if config.use_amp:
            with autocast():
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Métricas
        train_loss += loss.item()
        metrics = calculate_metrics(outputs, labels)
        train_correct += metrics['correct']
        train_total += metrics['total']
        
        # Log batch
        if (batch_idx + 1) % config.log_interval == 0:
            batch_acc = metrics['accuracy']
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} Acc: {batch_acc:.4f}")
    
    train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total
    
    # ===== VALIDATION PHASE =====
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(config.device), labels.to(config.device)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            metrics = calculate_metrics(outputs, labels)
            val_correct += metrics['correct']
            val_total += metrics['total']
    
    val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    
    # Learning rate atual
    current_lr = optimizer.param_groups[0]['lr']
    
    # Atualizar scheduler
    scheduler.step(val_loss)
    
    # Salvar histórico
    training_history['train_loss'].append(train_loss)
    training_history['train_acc'].append(train_acc)
    training_history['val_loss'].append(val_loss)
    training_history['val_acc'].append(val_acc)
    training_history['learning_rates'].append(current_lr)
    
    # TensorBoard logging
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    writer.add_scalar('Learning_Rate', current_lr, epoch)
    
    # Print epoch results
    print(f"\n  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
    print(f"  Learning Rate: {current_lr:.2e}")
    
    # Salvar checkpoint
    checkpoint_metrics = {
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'lr': current_lr
    }
    
    is_best = checkpoint_manager.save_checkpoint(
        epoch, model, optimizer, val_loss, val_acc, checkpoint_metrics
    )
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    
    # Early stopping check
    early_stopping(val_loss)
    if early_stopping.early_stop:
        print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
        break

# ===== SAVE FINAL MODEL =====
print("\n" + "="*50)
print("TRAINING COMPLETED")
print("="*50)

# Carregar melhor modelo
best_checkpoint = torch.load(os.path.join(config.checkpoint_dir, 'best_checkpoint.pth'))
model.load_state_dict(best_checkpoint['model_state_dict'])

# Salvar modelo final
final_model_path = "app/models/ai_detector_model.pth"
torch.save(model.state_dict(), final_model_path)
print(f"\n✓ Final model saved to: {final_model_path}")

# Salvar histórico de treinamento
history_path = os.path.join(config.checkpoint_dir, 'training_history.json')
with open(history_path, 'w') as f:
    json.dump(training_history, f, indent=2)
print(f"✓ Training history saved to: {history_path}")

# Print final statistics
print(f"\nBest Validation Accuracy: {best_val_acc:.4f}")
print(f"Best Validation Loss: {checkpoint_manager.best_val_loss:.4f}")
print(f"Total Epochs Trained: {epoch+1}")

writer.close()
print("\n✓ Training complete!")

