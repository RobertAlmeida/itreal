# Guia Rápido de Início

## 🚀 Início Rápido

### 1. Configuração Inicial

```bash
# Clone o repositório
git clone https://github.com/RobertAlmeida/itsreal.git
cd itsreal

# Copie o arquivo de configuração
cp .env.example .env

# Edite as configurações conforme necessário
nano .env
```

### 2. Instalação

#### Opção A: Instalação Local

```bash
# Crie ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Instale dependências
pip install -r requirements.txt
```

#### Opção B: Docker (Recomendado)

```bash
# Build e execute
docker-compose up -d

# Verifique status
docker-compose ps

# Veja logs
docker-compose logs -f api
```

### 3. Treinamento do Modelo

```bash
# Execute o treinamento
python train_model.py

# Visualize métricas no TensorBoard
tensorboard --logdir=runs

# Acesse: http://localhost:6006
```

### 4. Executar API

```bash
# Desenvolvimento
uvicorn app.main:app --reload

# Produção
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 5. Acessar Interface

- **Interface Web**: http://localhost:8000/index.html
- **Documentação API**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

---

## 📚 Recursos Principais

### Backend API

- ✅ **Rate Limiting**: Proteção contra abuso (configurável)
- ✅ **Cache**: Resultados armazenados para arquivos repetidos
- ✅ **Validação Robusta**: Tipo, tamanho, dimensões
- ✅ **Logging Estruturado**: Rastreamento completo
- ✅ **Health Checks**: Monitoramento de saúde

### Treinamento

- ✅ **Early Stopping**: Para automaticamente quando necessário
- ✅ **Checkpointing**: Salva melhores modelos
- ✅ **Mixed Precision**: Treinamento 2x mais rápido
- ✅ **TensorBoard**: Visualização de métricas
- ✅ **Data Augmentation**: Augmentations avançadas

---

## 🔧 Configuração

### Variáveis de Ambiente Principais

```bash
# API
DEBUG=False
LOG_LEVEL=INFO

# Limites de Upload
MAX_IMAGE_SIZE_MB=50
MAX_VIDEO_SIZE_MB=100

# Cache
ENABLE_CACHE=True
CACHE_TTL_SECONDS=3600

# Rate Limiting
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_PERIOD=60
```

---

## 📊 Endpoints da API

### GET /
Status da API

### GET /health
Health check detalhado
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cuda_available": false,
  "cache_enabled": true
}
```

### POST /detect/image
Analisa imagem
```bash
curl -X POST "http://localhost:8000/detect/image" \
  -F "file=@image.jpg"
```

### POST /detect/video
Analisa vídeo
```bash
curl -X POST "http://localhost:8000/detect/video" \
  -F "file=@video.mp4"
```

---

## 🐳 Docker

### Build Manual

```bash
# Build
docker build -t ai-detector .

# Run
docker run -p 8000:8000 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/app/models:/app/app/models \
  ai-detector
```

### Docker Compose

```bash
# Iniciar
docker-compose up -d

# Parar
docker-compose down

# Rebuild
docker-compose up -d --build

# Ver logs
docker-compose logs -f
```

---

## 🧪 Desenvolvimento

### Instalar Dependências de Dev

```bash
pip install -r requirements-dev.txt
```

### Code Quality

```bash
# Formatação
black app/

# Linting
flake8 app/

# Type checking
mypy app/

# Testes
pytest tests/ -v --cov=app
```

---

## 📈 Monitoramento

### Logs

```bash
# Ver logs em tempo real
tail -f logs/app.log

# Logs do Docker
docker-compose logs -f api
```

### Métricas de Treinamento

```bash
# TensorBoard
tensorboard --logdir=runs

# Acesse: http://localhost:6006
```

### Health Check

```bash
# Verificar saúde da API
curl http://localhost:8000/health

# Com jq para formatação
curl -s http://localhost:8000/health | jq
```

---

## 🔍 Troubleshooting

### Modelo não encontrado
```bash
# Verifique se o modelo existe
ls -lh app/models/ai_detector_model.pth

# Treine o modelo
python train_model.py
```

### Erro de CUDA
```bash
# Desabilite CUDA no .env
DEVICE=cpu
```

### Porta em uso
```bash
# Use porta diferente
uvicorn app.main:app --port 8001
```

### Docker build falha
```bash
# Limpe cache do Docker
docker system prune -a

# Rebuild sem cache
docker-compose build --no-cache
```

---

## 📝 Estrutura de Arquivos

```
itsreal/
├── app/
│   ├── config.py              # Configurações
│   ├── main.py                # API FastAPI
│   ├── models/
│   │   └── detector.py        # Modelo de detecção
│   ├── services/
│   │   ├── image_analyzer.py  # Análise de imagens
│   │   └── video_analyzer.py  # Análise de vídeos
│   └── utils/
│       ├── cache.py           # Sistema de cache
│       ├── logger.py          # Logging
│       └── validators.py      # Validações
├── checkpoints/               # Checkpoints de treinamento
├── dataset/                   # Dataset de treinamento
├── logs/                      # Logs da aplicação
├── runs/                      # TensorBoard logs
├── train_model.py            # Script de treinamento
├── index.html                # Interface web
├── Dockerfile                # Docker image
├── docker-compose.yml        # Docker orchestration
├── requirements.txt          # Dependências
└── .env.example              # Template de configuração
```

---

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Add nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abra um Pull Request

---

## 📄 Licença

MIT License - veja [LICENSE](LICENSE) para detalhes

---

## 👤 Autor

**Robert Almeida**
- GitHub: [@RobertAlmeida](https://github.com/RobertAlmeida)
- LinkedIn: [robertrochaalmeida](https://www.linkedin.com/in/robertrochaalmeida/)

---

## ⭐ Suporte

Se este projeto foi útil, considere dar uma estrela no GitHub!
