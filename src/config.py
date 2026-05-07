from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Set

from dotenv import load_dotenv


DEFAULT_DATABASE_URL = "postgresql://compatibility_user:compatibility_pass@localhost:5433/compatibility_bot"

# NLP defaults shared across scripts
NLP_DATA_PATH = "data/nlp_training_data.csv"
NLP_MODEL_PATH = "data/models/text_compatibility.joblib"
NLP_REPORT_PATH = "exports/text_compatibility_report.json"
NLP_MIN_EXAMPLES = 500
NLP_PRETRAINED_MODEL_NAME = os.getenv(
    "NLP_PRETRAINED_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
DEFAULT_LOG_FILE = "logs/bot.log"


@dataclass
class Settings:
    vk_token: str
    vk_group_id: int
    database_url: str
    admin_ids: Set[int]
    log_file: str
    log_level: str
def _parse_admin_ids(raw: str) -> Set[int]:
    result: Set[int] = set()
    for item in raw.replace(";", ",").split(","):
        value = item.strip()
        if not value:
            continue
        try:
            result.add(int(value))
        except ValueError:
            continue
    return result


# Собирает настройки приложения из переменных окружения.
def get_settings() -> Settings:
    load_dotenv()
    vk_token = os.getenv("VK_TOKEN", "").strip()
    try:
        vk_group_id = int(os.getenv("VK_GROUP_ID", "0").strip() or "0")
    except ValueError:
        vk_group_id = 0
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL).strip() or DEFAULT_DATABASE_URL
    admin_ids = _parse_admin_ids(os.getenv("ADMIN_IDS", ""))
    log_file = os.getenv("LOG_FILE", DEFAULT_LOG_FILE).strip() or DEFAULT_LOG_FILE
    log_level = os.getenv("LOG_LEVEL", "INFO").strip().upper() or "INFO"
    return Settings(
        vk_token=vk_token,
        vk_group_id=vk_group_id,
        database_url=database_url,
        admin_ids=admin_ids,
        log_file=log_file,
        log_level=log_level,
    )
