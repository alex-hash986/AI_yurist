"""Пути проекта и настройки для подключения к OpenAI."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "data" / "templates"
PRECEDENTS_DIR = BASE_DIR / "data" / "precedents"
SAMPLES_DIR = BASE_DIR / "data" / "sample_contracts"

load_dotenv(BASE_DIR / ".env")
CHROMA_PERSIST_DIR = (
    BASE_DIR / Path(os.getenv("CHROMA_PERSIST_DIR") or "chroma_db").expanduser()
).resolve()


@dataclass(frozen=True)
class Settings:
    """Параметры подключения к OpenAI."""

    openai_api_key: str
    openai_model: str


def load_settings() -> Settings:
    """Загрузить параметры OpenAI и проверить, что задан ключ API."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or api_key == "your_api_key_here":
        raise ValueError("Укажите действующий OPENAI_API_KEY в файле .env или в окружении.")

    return Settings(
        openai_api_key=api_key,
        openai_model=os.getenv("OPENAI_MODEL", "").strip() or "gpt-4o-mini",
    )
