import cv2
import tempfile
from PIL import Image
from fastapi import HTTPException

from app.utils.frame_utils import extract_frames
from app.models.detector import detector


async def analyze_video(file):
    try:
        # Salvar vídeo temporariamente
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(await file.read())
        temp.close()

        # Testar abertura do vídeo
        cap = cv2.VideoCapture(temp.name)
        if not cap.isOpened():
            raise HTTPException(
                status_code=422,
                detail="❌ O arquivo enviado não é um vídeo válido ou está corrompido."
            )
        cap.release()

        # Extrair frames com segurança
        try:
            frames = extract_frames(temp.name, every_n_frames=20)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"❌ Falha ao extrair frames do vídeo: {str(e)}"
            )

        if not frames or len(frames) == 0:
            raise HTTPException(
                status_code=422,
                detail="❌ Nenhum frame pôde ser extraído do vídeo. O arquivo pode estar corrompido ou em formato não suportado."
            )

        ai_scores = []

        for frame in frames:
            try:
                pil_frame = Image.fromarray(frame).convert("RGB")
                score = detector.predict(pil_frame)
                ai_scores.append(score["ai_probability"])
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Erro ao processar frame com modelo IA: {str(e)}"
                )

        avg_score = sum(ai_scores) / len(ai_scores)

        return {
            "type": "video",
            "frames_analyzed": len(ai_scores),
            "ai_probability": avg_score,
            "ai_probability_by_frame": ai_scores,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro inesperado ao analisar vídeo: {str(e)}"
        )
