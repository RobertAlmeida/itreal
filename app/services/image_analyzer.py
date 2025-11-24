from PIL import Image, UnidentifiedImageError
import tempfile
from fastapi import HTTPException
from app.utils.exif_utils import check_exif
from app.models.detector import detector

async def analyze_image(file):
    try:
        # Salvar temporariamente
        temp = tempfile.NamedTemporaryFile(delete=False)
        temp.write(await file.read())
        temp.close()

        # Tentativa de abrir como imagem
        try:
            image = Image.open(temp.name)
            image.verify()  # Verifica se é uma imagem válida
            image = Image.open(temp.name).convert("RGB")  # Recarrega corretamente
        except UnidentifiedImageError:
            raise HTTPException(
                status_code=422,
                detail="❌ O arquivo enviado não é uma imagem válida."
            )
        except Exception:
            raise HTTPException(
                status_code=422,
                detail="❌ Não foi possível processar a imagem enviada."
            )

        # 1. Verificar metadados
        try:
            exif_info = check_exif(temp.name)
        except Exception:
            exif_info = {
                "suspicious": True,
                "error": "Falha ao ler EXIF"
            }

        # 2. Rodar o modelo IA
        try:
            ai_score = detector.predict(image)
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Erro ao analisar imagem com o modelo IA: {str(e)}"
            )

        return {
            "type": "image",
            "ai_probability": ai_score,
            "metadata_suspicious": exif_info.get("suspicious", True),
            "exif": exif_info,
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro inesperado ao analisar imagem: {str(e)}"
        )
