🧠 ItsReal – Detector de Conteúdo Gerado por IA

Detecte se imagens ou vídeos foram criados por modelos de Inteligência Artificial usando redes neurais treinadas.
API construída em FastAPI, com backend em Python + PyTorch.

🚀 Funcionalidades

🔎 Detecção IA vs Real para imagens

🎥 Análise de vídeos com extração de frames

🧬 Modelo EfficientNet-B0 treinado

📝 Verificação de metadados EXIF

🔥 API FastAPI pronta para produção

⚡ Suporte a GPU (CUDA) quando disponível

🛡 Tratamento seguro de arquivos corrompidos

🗂 Estrutura do Projeto
itsreal/
│── app/
│   ├── main.py
│   ├── routes/
│   │   └── analyzer_routes.py
│   ├── services/
│   │   ├── image_analyzer.py
│   │   └── video_analyzer.py
│   ├── utils/
│   │   ├── exif_utils.py
│   │   └── frame_utils.py
│   ├── models/
│   │   ├── detector.py
│   │   └── ai_detector_model.pth  (IGNORADO NO GIT)
│── dataset/  (IGNORADO)
│── venv/     (IGNORADO)
│── .gitignore
│── README.md

🔧 Instalação
1️⃣ Clonar o repositório
git clone https://github.com/seuusuario/itsreal.git
cd itsreal

2️⃣ Criar ambiente virtual
python3 -m venv venv
source venv/bin/activate  # Linux/Mac

3️⃣ Instalar dependências
pip install -r requirements.txt

🤖 Treinando o Modelo

Coloque seu dataset no diretório:

dataset/
│── IA/
│── REAL/


Execute o script de treino:

python train.py


O modelo treinado será salvo automaticamente como:

ai_detector_model.pth


🔥 Importante: Arquivo ignorado no Git.

🧩 Rodando a API

Inicie o serviço FastAPI com Uvicorn:

uvicorn app.main:app --reload


A API estará disponível em:

👉 http://127.0.0.1:8000

👉 Documentação Swagger: http://127.0.0.1:8000/docs

📤 Endpoints
▶️ POST /analyze/image

Envia uma imagem para análise:

curl -X POST http://127.0.0.1:8000/analyze/image \
  -F "file=@foto.jpg"


Retorno:

{
  "type": "image",
  "ai_probability": {
    "ai_probability": 0.91,
    "real_probability": 0.09,
    "predicted": "IA"
  },
  "metadata_suspicious": true,
  "exif": {}
}

▶️ POST /analyze/video
curl -X POST http://127.0.0.1:8000/analyze/video \
  -F "file=@video.mp4"


Retorno:

{
  "type": "video",
  "frames_analyzed": 32,
  "ai_probability": 0.73,
  "ai_probability_by_frame": [...]
}

🛡 Segurança & Tratamento de Erros

Vídeos corrompidos → Erro claro

Fotos ilegíveis → Resposta com código 400

EXIF suspeito detectado

Limite automático de frames por vídeo

Risco de pickle mitigado (usar weights_only=True no futuro)

⚙️ Requisitos

Python 3.10+

PyTorch 2.x

CUDA 12+ (opcional)

OpenCV

FastAPI

📦 Roadmap

 Modelo de detecção multimodal (imagem + metadados)

 Dashboard admin

 Filtro anti-deepfake para rostos

 Suporte a vídeos longos (stream processing)

 Deploy em Docker/Kubernetes

🧑‍💻 Autor

Robert Almeida
Sistema de detecção de conteúdo com IA.

📜 Licença

MIT — livre para uso e modificação.