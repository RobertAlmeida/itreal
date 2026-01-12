"""
Serviço de análise de imagens para detecção de IA.
"""
from PIL import Image, UnidentifiedImageError
import tempfile
import os
from contextlib import contextmanager
from typing import Dict, Any
from fastapi import HTTPException, UploadFile

from app.config import settings
from app.utils.logger import app_logger
from app.utils.exif_utils import check_exif
from app.utils.validators import FileValidator, validate_and_save_temp_file
from app.utils.cache import cache_manager
from app.models.detector import detector


@contextmanager
def temporary_file(file: UploadFile):
    """Context manager para gerenciar arquivo temporário com cleanup automático"""
    temp_path = None
    try:
        temp_path = validate_and_save_temp_file(file)
        yield temp_path
    finally:
        # Cleanup automático
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
                app_logger.debug(f"Temporary file deleted: {temp_path}")
            except Exception as e:
                app_logger.warning(f"Failed to delete temporary file {temp_path}: {str(e)}")


async def analyze_image(file: UploadFile) -> Dict[str, Any]:
    """
    Analisa uma imagem e detecta se foi gerada por IA.
    
    Args:
        file: Arquivo de imagem enviado via upload
        
    Returns:
        Dict contendo:
            - type: "image"
            - ai_probability: Score de probabilidade de ser IA
            - metadata_suspicious: Boolean indicando metadados suspeitos
            - exif: Informações EXIF da imagem
            
    Raises:
        HTTPException: Em caso de erro na validação ou processamento
    """
    try:
        # Ler conteúdo do arquivo para cache
        file_content = await file.read()
        await file.seek(0)  # Reset para leitura posterior
        
        # Verificar cache
        cache_key = cache_manager.generate_key(file_content, "image")
        cached_result = cache_manager.get(cache_key)
        
        if cached_result:
            app_logger.info(f"Returning cached result for image: {file.filename}")
            return cached_result
        
        # Processar imagem com cleanup automático
        with temporary_file(file) as temp_path:
            # Abrir e validar imagem
            try:
                image = Image.open(temp_path)
                image.verify()  # Verifica integridade
                image = Image.open(temp_path).convert("RGB")  # Recarrega para uso
            except UnidentifiedImageError:
                app_logger.warning(f"Invalid image file: {file.filename}")
                raise HTTPException(
                    status_code=422,
                    detail="❌ O arquivo enviado não é uma imagem válida."
                )
            except Exception as e:
                app_logger.error(f"Error opening image: {str(e)}")
                raise HTTPException(
                    status_code=422,
                    detail="❌ Não foi possível processar a imagem enviada."
                )
            
            # Validar dimensões
            FileValidator.validate_image_dimensions(image)
            
            # 1. Verificar metadados EXIF
            try:
                exif_info = check_exif(temp_path)
                app_logger.debug(f"EXIF analysis completed: suspicious={exif_info.get('suspicious')}")
            except Exception as e:
                app_logger.warning(f"EXIF analysis failed: {str(e)}")
                exif_info = {
                    "suspicious": True,
                    "error": "Falha ao ler EXIF"
                }
            
            # 2. Executar modelo de detecção de IA
            try:
                ai_score = detector.predict(image)
                app_logger.info(
                    f"Image analysis completed: {file.filename} - "
                    f"AI probability: {ai_score.get('ai_probability', 'N/A')}"
                )
            except Exception as e:
                app_logger.error(f"Model prediction failed: {str(e)}", exc_info=True)
                raise HTTPException(
                    status_code=500,
                    detail=f"Erro ao analisar imagem com o modelo IA: {str(e)}"
                )
            
            # Preparar resultado
            result = {
                "type": "image",
                "filename": file.filename,
                "ai_probability": ai_score,
                "metadata_suspicious": exif_info.get("suspicious", True),
                "exif": exif_info,
            }
            
            # Armazenar em cache
            cache_manager.set(cache_key, result)
            
            return result

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Unexpected error analyzing image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erro inesperado ao analisar imagem: {str(e)}"
        )

