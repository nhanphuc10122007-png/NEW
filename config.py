import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from backend root or parent project root
backend_dir = Path(__file__).resolve().parent.parent.parent
env_path = backend_dir / ".env"
if not env_path.exists():
    env_path = backend_dir.parent / ".env"

load_dotenv(dotenv_path=env_path)

class Settings:
    PROJECT_NAME: str = "AURA BIOSTUDIO AI Backend"
    VERSION: str = "2.4.0"
    API_PREFIX: str = "/api"

    # CORS
    ALLOWED_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://localhost:8080",
        "*"
    ]

    # File uploads
    MAX_UPLOAD_SIZE: int = 15 * 1024 * 1024  # 15 MB
    ALLOWED_IMAGE_TYPES: list[str] = [
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
        "image/bmp"
    ]

    # Google Gemini API Key (Optional, for deep AI vision & NLP)
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()

    # Database
    DATABASE_PATH: Path = backend_dir / "biometric_history.db"

settings = Settings()
