"""Сбор реальных данных взаимодействий для переобучения NLP модели."""

from __future__ import annotations

import csv
import logging
import os
from datetime import datetime

from src.config import NLP_DATA_PATH, NLP_MIN_EXAMPLES


logger = logging.getLogger(__name__)


def log_interaction_for_nlp(
    db,
    viewer_id: int,
    viewed_id: int,
    interaction_type: str,
    output_csv: str = NLP_DATA_PATH
) -> None:
    """
    Логировать взаимодействие для переобучения.
    
    Args:
        db: Database объект
        viewer_id: ID пользователя, который совершил действие
        viewed_id: ID просмотренного профиля
        interaction_type: "like", "dislike", "block"
        output_csv: Путь для сохранения данных
    """
    
    try:
        viewer = db.get_user_profile(viewer_id)
        viewed = db.get_user_profile(viewed_id)
        
        if not viewer or not viewed:
            return
        
        # Получить тексты профилей: основной источник - текстовое поле профиля.
        text_left = str(getattr(viewer, "about_text", "") or viewer.questionnaire.get("about", "") or "")
        text_right = str(getattr(viewed, "about_text", "") or viewed.questionnaire.get("about", "") or "")
        
        text_left = text_left.strip()
        text_right = text_right.strip()
        if not text_left or not text_right:
            return
        
        # Конвертировать action в label
        label_map = {
            "like": "positive",
            "dislike": "negative",
            "block": "negative",
            "report": "negative",
        }
        label = label_map.get(interaction_type, "neutral")
        
        # Создать директорию
        output_dir = os.path.dirname(output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Добавить запись в CSV
        file_exists = os.path.exists(output_csv)
        with open(output_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "from_user_id", "to_user_id",
                "text_left", "text_right", "label"
            ])
            
            if not file_exists:
                writer.writeheader()
            
            writer.writerow({
                "timestamp": datetime.now().isoformat(),
                "from_user_id": viewer_id,
                "to_user_id": viewed_id,
                "text_left": text_left,
                "text_right": text_right,
                "label": label,
            })
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        # Ошибка логирования не должна помешать боту.
        logger.exception("Failed to collect interaction for NLP")


def get_nlp_stats(csv_path: str = NLP_DATA_PATH) -> dict:
    """Получить статистику собранных данных."""
    
    if not os.path.exists(csv_path):
        return {
            "total": 0,
            "positive": 0,
            "neutral": 0,
            "negative": 0,
            "ready": False
        }
    
    stats = {"total": 0, "positive": 0, "neutral": 0, "negative": 0}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stats["total"] += 1
            label = row.get("label", "").lower()
            if label in {"positive", "neutral", "negative"}:
                stats[label] += 1
    
    stats["ready"] = stats["total"] >= NLP_MIN_EXAMPLES
    return stats
