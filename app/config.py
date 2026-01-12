"""
Configuração centralizada da aplicação usando Pydantic Settings.
Gerencia variáveis de ambiente e constantes do sistema.
"""
from pydantic_settings import BaseSettings
from typing import List
import os


class Settings(BaseSettings):
    """Configurações da aplicação AI Media Detector"""
    
    # API Configuration
    app_name: str = "AI Media Detector API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # CORS Configuration
    cors_origins: List[str] = ["*"]
    cors_credentials: bool = True
    cors_methods: List[str] = ["*"]
    cors_headers: List[str] = ["*"]
    
    # File Upload Limits
    max_file_size_mb: int = 100
    max_image_size_mb: int = 50
    max_video_size_mb: int = 100
    max_video_duration_seconds: int = 300  # 5 minutos
    
    # Image Processing
    max_image_dimension: int = 4096
    min_image_dimension: int = 32
    
    # Video Processing
    video_frame_interval: int = 20
    max_frames_to_analyze: int = 200
    
    # Model Configuration
    model_path: str = "app/models/ai_detector_model.pth"
    device: str = "cuda"  # ou "cpu"
    
    # Cache Configuration
    enable_cache: bool = False
    cache_ttl_seconds: int = 3600  # 1 hora
    redis_url: str = "redis://localhost:6379"
    
    # Rate Limiting
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 10
    rate_limit_period: int = 60  # segundos
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"
    log_rotation: str = "10 MB"
    log_retention: str = "30 days"
    
    # Temporary Files
    temp_dir: str = "/tmp/ai_detector"
    auto_cleanup_temp_files: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Instância global de configurações
settings = Settings()


# Criar diretórios necessários
def setup_directories():
    """Cria diretórios necessários para a aplicação"""
    directories = [
        os.path.dirname(settings.log_file),
        settings.temp_dir,
        os.path.dirname(settings.model_path)
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


# Executar setup na importação
setup_directories()
