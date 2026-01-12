"""
Serviço de análise de vídeos para detecção de IA.
"""
import cv2
import os
from contextlib import contextmanager
from typing import Dict, Any, List
from PIL import Image
from fastapi import HTTPException, UploadFile

from app.config import settings
from app.utils.logger import app_logger
from app.utils.frame_utils import extract_frames
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


async def analyze_video(file: UploadFile) -> Dict[str, Any]:
    """
    Analisa um vídeo e detecta se foi gerado por IA.
    
    Args:
        file: Arquivo de vídeo enviado via upload
        
    Returns:
        Dict contendo:
            - type: "video"
            - frames_analyzed: Número de frames analisados
            - ai_probability: Probabilidade média de ser IA
            - ai_probability_by_frame: Lista de probabilidades por frame
            - duration: Duração do vídeo em segundos
            
    Raises:
        HTTPException: Em caso de erro na validação ou processamento
    """
    try:
        # Ler conteúdo do arquivo para cache
        file_content = await file.read()
        await file.seek(0)  # Reset para leitura posterior
        
        # Verificar cache
        cache_key = cache_manager.generate_key(file_content, "video")
        cached_result = cache_manager.get(cache_key)
        
        if cached_result:
            app_logger.info(f"Returning cached result for video: {file.filename}")
            return cached_result
        
        # Processar vídeo com cleanup automático
        with temporary_file(file) as temp_path:
            # Validar abertura do vídeo
            cap = cv2.VideoCapture(temp_path)
            if not cap.isOpened():
                app_logger.warning(f"Invalid video file: {file.filename}")
                raise HTTPException(
                    status_code=422,
                    detail="❌ O arquivo enviado não é um vídeo válido ou está corrompido."
                )
            cap.release()
            
            # Validar duração do vídeo
            try:
                duration = FileValidator.validate_video_duration(temp_path)
                app_logger.info(f"Video duration: {duration:.2f}s")
            except HTTPException:
                raise
            except Exception as e:
                app_logger.error(f"Error validating video duration: {str(e)}")
                raise HTTPException(
                    status_code=422,
                    detail=f"Erro ao validar duração do vídeo: {str(e)}"
                )
            
            # Extrair frames
            try:
                frames = extract_frames(
                    temp_path,
                    every_n_frames=settings.video_frame_interval,
                    max_frames=settings.max_frames_to_analyze
                )
                app_logger.info(f"Extracted {len(frames)} frames from video")
            except Exception as e:
                app_logger.error(f"Error extracting frames: {str(e)}")
                raise HTTPException(
                    status_code=422,
                    detail=f"❌ Falha ao extrair frames do vídeo: {str(e)}"
                )
            
            if not frames or len(frames) == 0:
                app_logger.warning(f"No frames extracted from video: {file.filename}")
                raise HTTPException(
                    status_code=422,
                    detail="❌ Nenhum frame pôde ser extraído do vídeo. "
                           "O arquivo pode estar corrompido ou em formato não suportado."
                )
            
            # Analisar frames
            ai_scores: List[float] = []
            
            for idx, frame in enumerate(frames):
                try:
                    pil_frame = Image.fromarray(frame).convert("RGB")
                    score = detector.predict(pil_frame)
                    ai_scores.append(score["ai_probability"])
                    
                    if (idx + 1) % 10 == 0:
                        app_logger.debug(f"Processed {idx + 1}/{len(frames)} frames")
                        
                except Exception as e:
                    app_logger.error(f"Error processing frame {idx}: {str(e)}")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Erro ao processar frame com modelo IA: {str(e)}"
                    )
            
            # Calcular média
            avg_score = sum(ai_scores) / len(ai_scores)
            
            app_logger.info(
                f"Video analysis completed: {file.filename} - "
                f"Frames: {len(ai_scores)}, Avg AI probability: {avg_score:.4f}"
            )
            
            # Preparar resultado
            result = {
                "type": "video",
                "filename": file.filename,
                "frames_analyzed": len(ai_scores),
                "ai_probability": avg_score,
                "ai_probability_by_frame": ai_scores,
                "duration": duration,
            }
            
            # Armazenar em cache
            cache_manager.set(cache_key, result)
            
            return result

    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Unexpected error analyzing video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erro inesperado ao analisar vídeo: {str(e)}"
        )

