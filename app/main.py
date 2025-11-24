from fastapi import FastAPI, UploadFile, File
from app.services.image_analyzer import analyze_image
from app.services.video_analyzer import analyze_video
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Media Detector API")


# Configuração CORS - Adicione ANTES das rotas
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens (em produção, especifique os domínios)
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, etc)
    allow_headers=["*"],  # Permite todos os headers
)


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    result = await analyze_image(file)
    return result


@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    print(file)
    result = await analyze_video(file)
    return result


@app.get("/")
def index():
    return {"status": "AI Detector API running"}
