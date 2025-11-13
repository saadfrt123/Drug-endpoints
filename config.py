import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load .env file if present (keeps defaults from actual environment)
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH, override=False)


def _split_origins(raw: str) -> List[str]:
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


@dataclass
class Settings:
    ENVIRONMENT: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "production"))
    DEBUG: bool = field(default_factory=lambda: os.getenv("DEBUG", "False").lower() == "true")

    NEO4J_URI: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687"))
    NEO4J_USER: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    NEO4J_PASSWORD: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))
    NEO4J_DATABASE: str = field(default_factory=lambda: os.getenv("NEO4J_DATABASE", "neo4j"))

    GEMINI_API_KEY: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    HOST: str = field(default_factory=lambda: os.getenv("HOST", "127.0.0.1"))
    PORT: int = field(default_factory=lambda: int(os.getenv("PORT", "8001")))

    API_KEY: str = field(default_factory=lambda: os.getenv("API_KEY", ""))
    ALLOWED_ORIGINS: List[str] = field(default_factory=lambda: _split_origins(os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")))

    LOG_LEVEL: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))


config = Settings()
