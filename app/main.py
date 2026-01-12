"""
AI Media Detector API - FastAPI Application
Detecta imagens e vídeos gerados por Inteligência Artificial
"""
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
from typing import Dict, Any

from app.config import settings
from app.utils.logger import app_logger, log_request, log_analysis
from app.services.image_analyzer import analyze_image
from app.services.video_analyzer import analyze_video
from app.utils.validators import FileValidator

# Inicializar rate limiter
limiter = Limiter(key_func=get_remote_address)

# Criar aplicação FastAPI
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API para detecção de imagens e vídeos gerados por IA usando Deep Learning",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Adicionar rate limiter à aplicação
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Configuração CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_credentials,
    allow_methods=settings.cors_methods,
    allow_headers=settings.cors_headers,
)


# Middleware para logging de requisições
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Middleware para logging de todas as requisições"""
    start_time = time.time()
    
    # Log da requisição
    log_request(
        endpoint=request.url.path,
        method=request.method,
        client=request.client.host if request.client else "unknown"
    )
    
    # Processar requisição
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        
        app_logger.info(
            f"Request completed - {request.method} {request.url.path} - "
            f"Status: {response.status_code} - Duration: {duration:.2f}s"
        )
        
        return response
    except Exception as e:
        duration = time.time() - start_time
        app_logger.error(
            f"Request failed - {request.method} {request.url.path} - "
            f"Duration: {duration:.2f}s - Error: {str(e)}"
        )
        raise


# Exception handler global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handler global para exceções não tratadas"""
    app_logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Erro interno do servidor",
            "error": str(exc) if settings.debug else "Internal server error"
        }
    )


@app.get("/", tags=["Health"])
async def root() -> Dict[str, str]:
    """
    Endpoint raiz - Verifica status da API
    """
    return {
        "status": "running",
        "app": settings.app_name,
        "version": settings.app_version
    }


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint - Verifica saúde da aplicação
    """
    import torch
    import os
    
    health_status = {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "model_loaded": os.path.exists(settings.model_path),
        "cuda_available": torch.cuda.is_available(),
        "cache_enabled": settings.enable_cache,
    }
    
    return health_status


@app.post("/detect/image", tags=["Detection"])
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_period}second")
async def detect_image_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Arquivo de imagem (JPG, PNG, WEBP)")
) -> Dict[str, Any]:
    """
    Analisa uma imagem e detecta se foi gerada por IA
    
    - **file**: Arquivo de imagem (JPG, PNG, WEBP)
    - **max_size**: 50MB
    - **returns**: Probabilidade de ser IA, metadados EXIF e análise
    """
    start_time = time.time()
    
    try:
        # Validar arquivo
        FileValidator.validate_image_file(file)
        
        # Analisar imagem
        result = await analyze_image(file)
        
        # Log da análise
        duration = time.time() - start_time
        log_analysis("image", file.size if hasattr(file, 'size') else 0, duration, result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error analyzing image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao analisar imagem: {str(e)}"
        )


@app.post("/detect/video", tags=["Detection"])
@limiter.limit(f"{settings.rate_limit_requests}/{settings.rate_limit_period}second")
async def detect_video_endpoint(
    request: Request,
    file: UploadFile = File(..., description="Arquivo de vídeo (MP4, MOV, AVI, WEBM)")
) -> Dict[str, Any]:
    """
    Analisa um vídeo e detecta se foi gerado por IA
    
    - **file**: Arquivo de vídeo (MP4, MOV, AVI, WEBM)
    - **max_size**: 100MB
    - **max_duration**: 5 minutos
    - **returns**: Probabilidade média de ser IA, análise por frame
    """
    start_time = time.time()
    
    try:
        # Validar arquivo
        FileValidator.validate_video_file(file)
        
        # Analisar vídeo
        result = await analyze_video(file)
        
        # Log da análise
        duration = time.time() - start_time
        log_analysis("video", file.size if hasattr(file, 'size') else 0, duration, result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error analyzing video: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao analisar vídeo: {str(e)}"
        )


# Startup event
@app.on_event("startup")
async def startup_event():
    """Executado ao iniciar a aplicação"""
    app_logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    app_logger.info(f"Debug mode: {settings.debug}")
    app_logger.info(f"Cache enabled: {settings.enable_cache}")
    app_logger.info(f"Rate limiting: {settings.rate_limit_enabled}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Executado ao desligar a aplicação"""
    app_logger.info(f"Shutting down {settings.app_name}")
    
    # Limpar cache se necessário
    from app.utils.cache import cache_manager
    cache_manager.clear()
