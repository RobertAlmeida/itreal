# 🧠 Guia Completo de Treinamento do Modelo

Este guia detalha todo o processo de treinamento do modelo de detecção de IA, desde a preparação dos dados até a avaliação final.

---

## 📋 Índice

1. [Pré-requisitos](#pré-requisitos)
2. [Preparação do Dataset](#preparação-do-dataset)
3. [Configuração do Ambiente](#configuração-do-ambiente)
4. [Execução do Treinamento](#execução-do-treinamento)
5. [Monitoramento](#monitoramento)
6. [Avaliação do Modelo](#avaliação-do-modelo)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Pré-requisitos

### Hardware Recomendado

- **GPU**: NVIDIA com CUDA (recomendado)
  - Mínimo: 6GB VRAM (GTX 1060, RTX 2060)
  - Recomendado: 8GB+ VRAM (RTX 3060, RTX 3070)
- **RAM**: 16GB mínimo, 32GB recomendado
- **Armazenamento**: 20GB+ livres

### Software

- Python 3.8+
- CUDA 11.8+ (se usar GPU)
- Git

### Verificar GPU (Opcional)

```bash
# Verificar se CUDA está disponível
python -c "import torch; print(f'CUDA disponível: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**Saída esperada com GPU**:
```
CUDA disponível: True
GPU: NVIDIA GeForce RTX 3060
```

**Saída esperada sem GPU**:
```
CUDA disponível: False
GPU: N/A
```

> ⚠️ **Nota**: O treinamento funciona em CPU, mas será **muito mais lento** (10-20x).

---

## 📁 Preparação do Dataset

### Estrutura de Diretórios

O dataset deve estar organizado assim:

```
dataset/
├── ai/          # Imagens geradas por IA
│   ├── img001.jpg
│   ├── img002.png
│   └── ...
└── real/        # Imagens reais/naturais
    ├── img001.jpg
    ├── img002.png
    └── ...
```

### Passo 1: Criar Estrutura

```bash
# Criar diretórios
mkdir -p dataset/ai
mkdir -p dataset/real
```

### Passo 2: Coletar Imagens

#### Imagens de IA (Classe `ai/`)

Fontes sugeridas:
- **Midjourney**: Imagens do Discord
- **DALL-E**: OpenAI
- **Stable Diffusion**: Geradores locais
- **Artbreeder**: Portraits gerados
- **ThisPersonDoesNotExist**: Rostos sintéticos

```bash
# Exemplo: baixar imagens de IA
cd dataset/ai/

# Adicione suas imagens aqui
# Formatos suportados: JPG, PNG, WEBP
```

#### Imagens Reais (Classe `real/`)

Fontes sugeridas:
- **Flickr**: Fotos reais com licença
- **Unsplash**: Fotos de alta qualidade
- **COCO Dataset**: Dataset público
- **ImageNet**: Subconjuntos
- **Suas próprias fotos**: Câmera/celular

```bash
# Exemplo: baixar imagens reais
cd dataset/real/

# Adicione suas imagens aqui
```

### Passo 3: Validar Dataset

```bash
# Contar imagens
echo "Imagens de IA: $(ls dataset/ai/ | wc -l)"
echo "Imagens reais: $(ls dataset/real/ | wc -l)"
```

**Recomendações**:
- ✅ **Mínimo**: 500 imagens por classe (1000 total)
- ✅ **Bom**: 1000-2000 imagens por classe
- ✅ **Ótimo**: 5000+ imagens por classe
- ✅ **Balanceamento**: Número similar em ambas as classes

### Passo 4: Verificar Qualidade

```python
# Script para verificar imagens corrompidas
from PIL import Image
import os

def check_images(directory):
    corrupted = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        try:
            img = Image.open(filepath)
            img.verify()
        except Exception as e:
            corrupted.append(filepath)
            print(f"❌ Corrompida: {filepath}")
    
    print(f"\n✅ Total verificadas: {len(os.listdir(directory))}")
    print(f"❌ Corrompidas: {len(corrupted)}")
    return corrupted

# Verificar
print("Verificando imagens de IA...")
check_images("dataset/ai")

print("\nVerificando imagens reais...")
check_images("dataset/real")
```

---

## ⚙️ Configuração do Ambiente

### Passo 1: Criar Ambiente Virtual

```bash
# Criar venv
python -m venv venv

# Ativar
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

### Passo 2: Instalar Dependências

```bash
# Atualizar pip
pip install --upgrade pip

# Instalar dependências
pip install -r requirements.txt
```

**Tempo estimado**: 5-10 minutos

### Passo 3: Configurar Parâmetros de Treinamento

Edite `train_model.py` se necessário:

```python
class TrainingConfig:
    # Dados
    dataset_path = "./dataset"
    val_ratio = 0.2              # 20% para validação
    batch_size = 32              # Reduzir se pouca VRAM
    
    # Treinamento
    epochs = 150                 # Máximo de epochs
    learning_rate = 1e-5         # Taxa de aprendizado
    
    # Early Stopping
    early_stopping_patience = 15 # Parar após 15 epochs sem melhora
    
    # Mixed Precision
    use_amp = True               # Usar AMP (mais rápido)
    
    # Device
    device = "cuda"              # "cuda" ou "cpu"
```

**Ajustes comuns**:

| Situação | Ajuste |
|----------|--------|
| Pouca VRAM (4-6GB) | `batch_size = 16` |
| Muito pouca VRAM (<4GB) | `batch_size = 8`, `use_amp = False` |
| CPU apenas | `device = "cpu"`, `batch_size = 16` |
| Dataset pequeno (<1000) | `epochs = 100`, `early_stopping_patience = 10` |

---

## 🚀 Execução do Treinamento

### Passo 1: Iniciar Treinamento

```bash
# Executar script
python train_model.py
```

### Passo 2: Entender a Saída

**Início do treinamento**:
```
Loading dataset...
Total images: 2000
Classes: ['ai', 'real']
Training images: 1600
Validation images: 400

Initializing efficientnet_b0...
Model loaded on cuda

==================================================
STARTING TRAINING
==================================================
```

**Durante cada epoch**:
```
Epoch 1/150
--------------------------------------------------
  Batch [10/50] Loss: 0.6234 Acc: 0.6250
  Batch [20/50] Loss: 0.5891 Acc: 0.6875
  ...

  Train Loss: 0.5234 | Train Acc: 0.7125
  Val Loss:   0.4891 | Val Acc:   0.7625
  Learning Rate: 1.00e-05
  
✓ Best checkpoint saved! Val Loss: 0.4891
```

**Métricas importantes**:
- `Train Loss`: Menor é melhor (objetivo: <0.3)
- `Train Acc`: Maior é melhor (objetivo: >0.90)
- `Val Loss`: Menor é melhor (objetivo: <0.4)
- `Val Acc`: Maior é melhor (objetivo: >0.85)

### Passo 3: Early Stopping

Se o modelo parar de melhorar:

```
EarlyStopping counter: 1/15
EarlyStopping counter: 2/15
...
EarlyStopping counter: 15/15

⚠️  Early stopping triggered at epoch 45
```

Isso é **normal** e **desejável** - previne overfitting!

### Passo 4: Conclusão

```
==================================================
TRAINING COMPLETED
==================================================

✓ Final model saved to: app/models/ai_detector_model.pth
✓ Training history saved to: checkpoints/training_history.json

Best Validation Accuracy: 0.8750
Best Validation Loss: 0.3245
Total Epochs Trained: 45

✓ Training complete!
```

**Tempo estimado**:
- **GPU (RTX 3060)**: 30-60 minutos
- **GPU (GTX 1060)**: 1-2 horas
- **CPU**: 8-12 horas

---

## 📊 Monitoramento

### Opção 1: TensorBoard (Recomendado)

**Terminal 1** (Treinamento):
```bash
python train_model.py
```

**Terminal 2** (TensorBoard):
```bash
# Iniciar TensorBoard
tensorboard --logdir=runs

# Acesse: http://localhost:6006
```

**Gráficos disponíveis**:
- 📉 Loss/train - Perda no treino
- 📉 Loss/val - Perda na validação
- 📈 Accuracy/train - Acurácia no treino
- 📈 Accuracy/val - Acurácia na validação
- 📊 Learning_Rate - Taxa de aprendizado

**O que observar**:
- ✅ **Bom**: Val loss diminuindo, val acc aumentando
- ⚠️ **Overfitting**: Train acc >> Val acc (diferença >10%)
- ❌ **Underfitting**: Ambas as acurácias baixas (<70%)

### Opção 2: Logs em Tempo Real

```bash
# Em outro terminal
tail -f logs/app.log
```

### Opção 3: Histórico JSON

Após o treinamento:

```bash
# Ver histórico
cat checkpoints/training_history.json | jq
```

---

## 🎯 Avaliação do Modelo

### Verificar Checkpoints

```bash
# Listar checkpoints salvos
ls -lh checkpoints/

# Saída esperada:
# best_checkpoint.pth      - Melhor modelo
# last_checkpoint.pth      - Último epoch
# training_history.json    - Histórico completo
```

### Testar Modelo

```bash
# Iniciar API
uvicorn app.main:app --reload

# Em outro terminal, testar com imagem
curl -X POST "http://localhost:8000/detect/image" \
  -F "file=@test_image.jpg"
```

**Resposta esperada**:
```json
{
  "type": "image",
  "filename": "test_image.jpg",
  "ai_probability": {
    "ai_probability": 0.8234,
    "real_probability": 0.1766,
    "predicted": "IA"
  },
  "metadata_suspicious": true,
  "exif": {...}
}
```

### Métricas de Qualidade

**Excelente modelo**:
- ✅ Val Accuracy > 90%
- ✅ Val Loss < 0.3
- ✅ Diferença Train/Val Acc < 5%

**Bom modelo**:
- ✅ Val Accuracy > 85%
- ✅ Val Loss < 0.4
- ✅ Diferença Train/Val Acc < 10%

**Modelo aceitável**:
- ⚠️ Val Accuracy > 75%
- ⚠️ Val Loss < 0.5
- ⚠️ Diferença Train/Val Acc < 15%

**Modelo ruim** (retreinar):
- ❌ Val Accuracy < 75%
- ❌ Val Loss > 0.5
- ❌ Diferença Train/Val Acc > 15%

---

## 🔍 Troubleshooting

### Erro: CUDA out of memory

**Sintoma**:
```
RuntimeError: CUDA out of memory
```

**Solução**:
```python
# Editar train_model.py
class TrainingConfig:
    batch_size = 16  # ou 8
    use_amp = True   # Certifique-se que está True
```

### Erro: Dataset vazio

**Sintoma**:
```
RuntimeError: Found 0 files in subfolders of: ./dataset
```

**Solução**:
```bash
# Verificar estrutura
ls -R dataset/

# Deve ter:
# dataset/ai/
# dataset/real/
```

### Overfitting (Train Acc >> Val Acc)

**Sintoma**:
- Train Acc: 95%
- Val Acc: 70%

**Soluções**:
1. **Mais dados**: Adicionar mais imagens
2. **Mais augmentation**: Editar `train_transform`
3. **Early stopping**: Já implementado
4. **Regularização**: Aumentar `weight_decay`

```python
class TrainingConfig:
    weight_decay = 1e-3  # Aumentar de 1e-4
```

### Underfitting (Ambas acurácias baixas)

**Sintoma**:
- Train Acc: 65%
- Val Acc: 63%

**Soluções**:
1. **Mais epochs**: Aumentar `epochs`
2. **Learning rate**: Aumentar para `1e-4`
3. **Modelo maior**: Usar `efficientnet_b1` ou `b2`

### Treinamento muito lento

**CPU**:
```python
# Reduzir batch size
batch_size = 8
num_workers = 2
```

**GPU antiga**:
```python
# Desabilitar AMP
use_amp = False
batch_size = 16
```

### Modelo não melhora

**Sintoma**:
- Val Loss estagnado em ~0.69 (50% accuracy)

**Causas possíveis**:
1. **Dataset ruim**: Imagens muito similares
2. **Learning rate alto**: Reduzir para `1e-6`
3. **Modelo congelado**: Verificar `requires_grad=True`

---

## 📈 Melhorando o Modelo

### 1. Aumentar Dataset

Mais dados = melhor modelo!

**Objetivo**: 5000+ imagens por classe

### 2. Data Augmentation Customizada

Editar `train_transform` em `train_model.py`:

```python
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(20),  # Aumentar rotação
    transforms.RandomResizedCrop(256, scale=(0.6, 1.0)),  # Mais crop
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
    transforms.RandomGrayscale(p=0.15),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),  # Adicionar blur
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### 3. Experimentar Modelos Maiores

```python
# Em train_model.py
model = models.efficientnet_b1(pretrained=True)  # ou b2, b3
```

**Trade-off**:
- ✅ Maior acurácia
- ❌ Mais lento
- ❌ Mais VRAM

### 4. Learning Rate Finder

```python
# Testar diferentes learning rates
learning_rates = [1e-6, 5e-6, 1e-5, 5e-5, 1e-4]

# Treinar por 5 epochs cada e comparar
```

---

## ✅ Checklist Final

Antes de usar o modelo em produção:

- [ ] Val Accuracy > 85%
- [ ] Val Loss < 0.4
- [ ] Testado com imagens reais
- [ ] Testado com imagens de IA
- [ ] Sem overfitting (Train/Val diff < 10%)
- [ ] Checkpoints salvos
- [ ] Histórico de treinamento documentado
- [ ] TensorBoard logs revisados

---

## 📚 Recursos Adicionais

### Datasets Públicos

- **CIFAKE**: https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images
- **AI vs Real**: https://www.kaggle.com/datasets/superpotato9/dalle-recognition-dataset
- **Synthetic Faces**: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

### Ferramentas

- **TensorBoard**: Visualização de métricas
- **Weights & Biases**: Tracking de experimentos
- **MLflow**: Gerenciamento de modelos

### Leitura Recomendada

- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Transfer Learning Guide](https://cs231n.github.io/transfer-learning/)
- [Data Augmentation Techniques](https://pytorch.org/vision/stable/transforms.html)

---

## 🎓 Próximos Passos

Após treinar o modelo com sucesso:

1. **Deploy**: Usar Docker para produção
2. **Monitoramento**: Configurar logs e métricas
3. **A/B Testing**: Comparar versões do modelo
4. **Retreinamento**: Atualizar com novos dados periodicamente

---

**Boa sorte com o treinamento! 🚀**

Se tiver dúvidas, consulte o [README](README.md) ou abra uma issue no GitHub.
