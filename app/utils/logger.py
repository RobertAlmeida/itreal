"""
Sistema de logging estruturado para a aplicação.
Usa loguru para logging simplificado e poderoso.
"""
from loguru import logger
import sys
from app.config import settings


def setup_logger():
    """Configura o sistema de logging da aplicação"""
    
    # Remove handlers padrão
    logger.remove()
    
    # Console handler com cores
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=settings.log_level,
        colorize=True,
    )
    
    # File handler com rotação
    logger.add(
        settings.log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=settings.log_level,
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression="zip",
        enqueue=True,  # Thread-safe
    )
    
    logger.info(f"Logger configurado - Level: {settings.log_level}")
    return logger


# Configurar logger na importação
app_logger = setup_logger()


def log_request(endpoint: str, method: str, client: str):
    """Log de requisições HTTP"""
    app_logger.info(f"Request: {method} {endpoint} from {client}")


def log_analysis(file_type: str, file_size: int, duration: float, result: dict):
    """Log de análises realizadas"""
    app_logger.info(
        f"Analysis completed - Type: {file_type}, Size: {file_size}B, "
        f"Duration: {duration:.2f}s, AI Probability: {result.get('ai_probability', 'N/A')}"
    )


def log_error(error: Exception, context: dict = None):
    """Log de erros com contexto"""
    app_logger.error(f"Error: {str(error)}", extra=context or {})
    app_logger.exception(error)
