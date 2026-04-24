from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


# NLP defaults shared across scripts
NLP_DATA_PATH = "data/nlp_training_data.csv"
NLP_MODEL_PATH = "models/nlp_model.pkl"
NLP_REPORT_PATH = "reports/nlp_latest_report.json"
NLP_MIN_EXAMPLES = 500


@dataclass
class Settings:
    vk_token: str
    database_url: str


def get_settings() -> Settings:
    load_dotenv()
    vk_token = os.getenv("VK_TOKEN", "").strip()
    database_url = os.getenv("DATABASE_URL", "").strip()
    return Settings(vk_token=vk_token, database_url=database_url)
