"""
Validadores para arquivos de upload e parâmetros da API.
"""
from fastapi import HTTPException, UploadFile
from PIL import Image
import cv2
import tempfile
import os
from typing import Tuple
from app.config import settings
from app.utils.logger import app_logger


class FileValidator:
    """Validador de arquivos de upload"""
    
    ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
    ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm'}
    
    ALLOWED_IMAGE_MIMES = {
        'image/jpeg', 'image/png', 'image/webp', 'image/jpg'
    }
    ALLOWED_VIDEO_MIMES = {
        'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm'
    }
    
    @staticmethod
    def validate_file_size(file: UploadFile, max_size_mb: int = None) -> None:
        """Valida tamanho do arquivo"""
        max_size = max_size_mb or settings.max_file_size_mb
        max_bytes = max_size * 1024 * 1024
        
        # Ler tamanho do arquivo
        file.file.seek(0, 2)  # Ir para o final
        file_size = file.file.tell()
        file.file.seek(0)  # Voltar ao início
        
        if file_size > max_bytes:
            app_logger.warning(f"File too large: {file_size} bytes (max: {max_bytes})")
            raise HTTPException(
                status_code=413,
                detail=f"Arquivo muito grande. Tamanho máximo: {max_size}MB"
            )
        
        app_logger.debug(f"File size validated: {file_size} bytes")
    
    @staticmethod
    def validate_image_file(file: UploadFile) -> None:
        """Valida arquivo de imagem"""
        # Validar extensão
        filename = file.filename.lower()
        ext = os.path.splitext(filename)[1]
        
        if ext not in FileValidator.ALLOWED_IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Extensão de imagem não suportada: {ext}. "
                       f"Permitidas: {', '.join(FileValidator.ALLOWED_IMAGE_EXTENSIONS)}"
            )
        
        # Validar MIME type
        if file.content_type not in FileValidator.ALLOWED_IMAGE_MIMES:
            raise HTTPException(
                status_code=422,
                detail=f"Tipo de arquivo não suportado: {file.content_type}"
            )
        
        # Validar tamanho
        FileValidator.validate_file_size(file, settings.max_image_size_mb)
        
        app_logger.info(f"Image file validated: {filename}")
    
    @staticmethod
    def validate_video_file(file: UploadFile) -> None:
        """Valida arquivo de vídeo"""
        # Validar extensão
        filename = file.filename.lower()
        ext = os.path.splitext(filename)[1]
        
        if ext not in FileValidator.ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(
                status_code=422,
                detail=f"Extensão de vídeo não suportada: {ext}. "
                       f"Permitidas: {', '.join(FileValidator.ALLOWED_VIDEO_EXTENSIONS)}"
            )
        
        # Validar MIME type
        if file.content_type not in FileValidator.ALLOWED_VIDEO_MIMES:
            raise HTTPException(
                status_code=422,
                detail=f"Tipo de arquivo não suportado: {file.content_type}"
            )
        
        # Validar tamanho
        FileValidator.validate_file_size(file, settings.max_video_size_mb)
        
        app_logger.info(f"Video file validated: {filename}")
    
    @staticmethod
    def validate_image_dimensions(image: Image.Image) -> None:
        """Valida dimensões da imagem"""
        width, height = image.size
        
        if width < settings.min_image_dimension or height < settings.min_image_dimension:
            raise HTTPException(
                status_code=422,
                detail=f"Imagem muito pequena. Dimensão mínima: {settings.min_image_dimension}px"
            )
        
        if width > settings.max_image_dimension or height > settings.max_image_dimension:
            raise HTTPException(
                status_code=422,
                detail=f"Imagem muito grande. Dimensão máxima: {settings.max_image_dimension}px"
            )
        
        app_logger.debug(f"Image dimensions validated: {width}x{height}")
    
    @staticmethod
    def validate_video_duration(video_path: str) -> float:
        """Valida duração do vídeo e retorna a duração em segundos"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise HTTPException(
                status_code=422,
                detail="Não foi possível abrir o vídeo"
            )
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        if fps <= 0:
            fps = 30  # Fallback
        
        duration = frame_count / fps
        
        if duration > settings.max_video_duration_seconds:
            raise HTTPException(
                status_code=422,
                detail=f"Vídeo muito longo. Duração máxima: {settings.max_video_duration_seconds}s"
            )
        
        app_logger.debug(f"Video duration validated: {duration:.2f}s")
        return duration


def validate_and_save_temp_file(file: UploadFile) -> str:
    """Valida e salva arquivo temporário de forma segura"""
    try:
        # Criar arquivo temporário
        temp = tempfile.NamedTemporaryFile(delete=False, dir=settings.temp_dir)
        
        # Ler e escrever conteúdo
        content = file.file.read()
        temp.write(content)
        temp.close()
        
        # Resetar posição do arquivo original
        file.file.seek(0)
        
        app_logger.debug(f"Temporary file created: {temp.name}")
        return temp.name
        
    except Exception as e:
        app_logger.error(f"Error saving temporary file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erro ao processar arquivo"
        )
