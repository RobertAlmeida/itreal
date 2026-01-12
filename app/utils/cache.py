"""
Sistema de cache simples para resultados de análise.
Suporta cache em memória e Redis (opcional).
"""
import hashlib
import json
from typing import Optional, Any
from datetime import datetime, timedelta
from app.config import settings
from app.utils.logger import app_logger


class InMemoryCache:
    """Cache simples em memória com TTL"""
    
    def __init__(self):
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Recupera valor do cache se não expirado"""
        if key not in self._cache:
            return None
        
        # Verificar expiração
        if key in self._timestamps:
            if datetime.now() > self._timestamps[key]:
                # Expirado, remover
                del self._cache[key]
                del self._timestamps[key]
                return None
        
        app_logger.debug(f"Cache hit: {key}")
        return self._cache[key]
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Armazena valor no cache com TTL"""
        ttl = ttl or settings.cache_ttl_seconds
        self._cache[key] = value
        self._timestamps[key] = datetime.now() + timedelta(seconds=ttl)
        app_logger.debug(f"Cache set: {key} (TTL: {ttl}s)")
    
    def delete(self, key: str):
        """Remove valor do cache"""
        if key in self._cache:
            del self._cache[key]
        if key in self._timestamps:
            del self._timestamps[key]
        app_logger.debug(f"Cache delete: {key}")
    
    def clear(self):
        """Limpa todo o cache"""
        self._cache.clear()
        self._timestamps.clear()
        app_logger.info("Cache cleared")


class CacheManager:
    """Gerenciador de cache com suporte para diferentes backends"""
    
    def __init__(self):
        self.enabled = settings.enable_cache
        self._backend = InMemoryCache()
        app_logger.info(f"Cache initialized - Enabled: {self.enabled}")
    
    def generate_key(self, file_content: bytes, file_type: str) -> str:
        """Gera chave única baseada no conteúdo do arquivo"""
        hash_obj = hashlib.sha256(file_content)
        return f"{file_type}:{hash_obj.hexdigest()}"
    
    def get(self, key: str) -> Optional[dict]:
        """Recupera resultado do cache"""
        if not self.enabled:
            return None
        return self._backend.get(key)
    
    def set(self, key: str, value: dict, ttl: int = None):
        """Armazena resultado no cache"""
        if not self.enabled:
            return
        self._backend.set(key, value, ttl)
    
    def delete(self, key: str):
        """Remove resultado do cache"""
        if not self.enabled:
            return
        self._backend.delete(key)
    
    def clear(self):
        """Limpa todo o cache"""
        if not self.enabled:
            return
        self._backend.clear()


# Instância global do cache
cache_manager = CacheManager()
