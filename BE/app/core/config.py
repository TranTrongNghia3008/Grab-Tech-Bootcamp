from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str
    
    data_storage_dir: str = 'data'
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        
settings = Settings()