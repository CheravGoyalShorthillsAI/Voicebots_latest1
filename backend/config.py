# Configuration management for the Voicebot application
import os
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv(override=True)

class Config:
    """Application configuration management"""
    
    # API Keys
    DEEPGRAM_API_KEY: str = os.getenv('DEEPGRAM_API_KEY', '')
    AZURE_OPENAI_API_KEY: str = os.getenv('AZURE_OPENAI_API_KEY', '')
    AZURE_OPENAI_ENDPOINT: str = os.getenv('AZURE_OPENAI_ENDPOINT', '')
    AZURE_OPENAI_DEPLOYMENT_NAME: str = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', '')
    AZURE_OPENAI_API_VERSION: str = os.getenv('AZURE_OPENAI_API_VERSION', '')
    AWS_ACCESS_KEY_ID: str = os.getenv('AWS_ACCESS_KEY_ID', '')
    AWS_SECRET_ACCESS_KEY: str = os.getenv('AWS_SECRET_ACCESS_KEY', '')
    
    # Audio Configuration
    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1
    CHUNK_SIZE: int = 1024
    AUDIO_FORMAT: str = "linear16"
    
    # LLM Configuration
    LLM_MODEL: str = "nova-3"
    LANGUAGE: str = "en-US"
    MAX_TOKENS: int = 500
    TEMPERATURE: float = 0.7
    
    # Application Settings
    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    LOG_LEVEL: str = os.getenv('LOG_LEVEL', 'INFO')
    CONVERSATIONS_FILE: str = os.getenv('CONVERSATIONS_FILE', 'conversations.json')
    MAX_CONVERSATION_HISTORY: int = int(os.getenv('MAX_CONVERSATION_HISTORY', '10'))
    
    # Health Check Settings
    HEALTH_CHECK_ENABLED: bool = True
    METRICS_ENABLED: bool = True
    
    @classmethod
    def validate_config(cls) -> list[str]:
        """Validate configuration and return list of missing required fields"""
        missing_fields = []
        
        required_fields = [
            ('DEEPGRAM_API_KEY', cls.DEEPGRAM_API_KEY),
            ('AZURE_OPENAI_API_KEY', cls.AZURE_OPENAI_API_KEY),
            ('AZURE_OPENAI_ENDPOINT', cls.AZURE_OPENAI_ENDPOINT),
            ('AWS_ACCESS_KEY_ID', cls.AWS_ACCESS_KEY_ID),
            ('AWS_SECRET_ACCESS_KEY', cls.AWS_SECRET_ACCESS_KEY),
        ]
        
        for field_name, field_value in required_fields:
            if not field_value:
                missing_fields.append(field_name)
        
        return missing_fields
    
    @classmethod
    def setup_logging(cls):
        """Configure application logging"""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('voicebot.log')
            ]
        )

# Global config instance
config = Config()

# Validate configuration on import
missing_config = config.validate_config()
if missing_config:
    logging.warning(f"Missing configuration fields: {', '.join(missing_config)}")