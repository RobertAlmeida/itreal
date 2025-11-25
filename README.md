# AI Media Detector API

API FastAPI para detecção de imagens e vídeos gerados por Inteligência Artificial usando Deep Learning.

## 📋 Sobre

Este projeto utiliza um modelo baseado em EfficientNet-B0 para classificar imagens e vídeos como **gerados por IA** ou **reais**. A API também analisa metadados EXIF para identificar possíveis manipulações.

## 🚀 Funcionalidades

- **Detecção em Imagens**: Analisa imagens individuais e retorna probabilidade de ser IA
- **Detecção em Vídeos**: Processa frames de vídeos para análise
- **Análise de Metadados**: Verifica EXIF para detectar ausência ou inconsistências
- **CORS Habilitado**: Pronto para integração com front-end
- **Interface Web**: Página HTML incluída para testes

## 📁 Estrutura do Projeto

```
app/
├── models/
│   ├── detector.py           # Modelo de Deep Learning
│   └── ai_detector_model.pth # Pesos treinados (você precisa adicionar)
├── schemas/
├── services/
│   ├── image_analyzer.py     # Análise de imagens
│   └── video_analyzer.py     # Análise de vídeos
└── utils/
    ├── exif_utils.py         # Utilitários para metadados
    └── frame_utils.py        # Extração de frames

main.py                       # Aplicação FastAPI
index.html                    # Interface web
```

## 🔧 Instalação

### Pré-requisitos

- Python 3.8+
- pip

### Passos

1. **Clone o repositório**
```bash
git clone https://github.com/RobertAlmeida/ai-media-detector.git
cd ai-media-detector
```

2. **Instale as dependências**
```bash
pip install fastapi uvicorn python-multipart pillow torch torchvision
```

3. **Adicione o modelo treinado**

Coloque o arquivo `ai_detector_model.pth` dentro da pasta `app/models/`:
```
app/models/ai_detector_model.pth
```

4. **Execute a API**
```bash
uvicorn main:app --reload
```

A API estará disponível em: `http://localhost:8000`

## 📡 Endpoints

### `GET /`
Verifica status da API

**Resposta:**
```json
{
  "status": "AI Detector API running"
}
```

### `POST /detect/image`
Analisa uma imagem

**Parâmetros:**
- `file` (multipart/form-data): Arquivo de imagem

**Resposta:**
```json
{
  "type": "image",
  "ai_probability": {
    "ai_probability": 0.8410249352455139,
    "real_probability": 0.15897512435913086,
    "predicted": "IA"
  },
  "metadata_suspicious": true,
  "exif": {
    "suspicious": true,
    "reason": "EXIF missing",
    "tags": {}
  }
}
```

### `POST /detect/video`
Analisa um vídeo

**Parâmetros:**
- `file` (multipart/form-data): Arquivo de vídeo

**Resposta:** Estrutura similar à detecção de imagem

## 🖥️ Interface Web

Abra o arquivo `index.html` no navegador para usar a interface visual. Certifique-se de que a API está rodando antes de fazer uploads.

## 🧠 Modelo

O detector usa **EfficientNet-B0** com:
- Entrada: Imagens 256x256 pixels
- Saída: 2 classes (IA / REAL)
- Framework: PyTorch
- Arquitetura modificada para classificação binária

## 🛠️ Tecnologias

- **FastAPI**: Framework web moderno e rápido
- **PyTorch**: Deep Learning
- **Torchvision**: Transformações de imagem
- **Pillow**: Processamento de imagens
- **CORS Middleware**: Integração front-end

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.

## 👤 Autor

**Robert Almeida**

- GitHub: [@RobertAlmeida](https://github.com/RobertAlmeida)
- LinkedIn: [robertrochaalmeida](https://www.linkedin.com/in/robertrochaalmeida/)

## 🤝 Contribuindo

Contribuições são bem-vindas! Sinta-se à vontade para:

1. Fazer um Fork do projeto
2. Criar uma branch para sua feature (`git checkout -b feature/nova-feature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/nova-feature`)
5. Abrir um Pull Request

## ⚠️ Notas Importantes

- O arquivo `ai_detector_model.pth` **não está incluído** no repositório
- Você precisa treinar ou obter um modelo compatível com a arquitetura EfficientNet-B0
- Para produção, configure `allow_origins` no CORS com domínios específicos
- Considere adicionar autenticação para uso em produção

## 📊 Exemplo de Uso com cURL

```bash
# Testar imagem
curl -X POST "http://localhost:8000/detect/image" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sua_imagem.jpg"

# Testar vídeo
curl -X POST "http://localhost:8000/detect/video" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@seu_video.mp4"
```

## 🐛 Troubleshooting

### Erro: "Modelo não encontrado"
- Verifique se `ai_detector_model.pth` está em `app/models/`
- Confirme que o caminho está correto

### Erro de CORS
- Verifique se o middleware CORS está configurado antes das rotas
- Em produção, especifique os domínios permitidos

### Erro de memória com vídeos grandes
- Considere processar vídeos em batches menores
- Aumente a memória disponível ou reduza a resolução dos frames

---

⭐ Se este projeto foi útil, considere dar uma estrela no GitHub!