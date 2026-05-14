from __future__ import annotations

import csv
import json
import os
from statistics import mean
from typing import Dict, List, Optional

from src.database import Database


NLP_EXPORT_FIELDS = [
    "feedback_id",
    "from_user_id",
    "to_user_id",
    "from_about",
    "to_about",
    "from_answers_json",
    "to_answers_json",
    "liked",
    "meeting_agree",
    "user_score",
    "label",
    "created_at",
]
def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# Собирает агрегированную статистику отзывов.
def collect_feedback_stats(db: Database, user_id: Optional[int] = None) -> Dict[str, float]:
    rows = db.get_feedback_rows_by_author(user_id) if user_id is not None else db.get_feedback_rows()
    if not rows:
        return {
            "avg_user_score": 0.0,
            "likes_count": 0.0,
            "meetings_agreed": 0.0,
            "successful_matches_ratio": 0.0,
        }

    scores: List[int] = [r["user_score"] for r in rows if r["user_score"] is not None]
    likes = sum(int(r["liked"]) for r in rows)
    meetings = sum(int(r["meeting_agree"]) for r in rows)
    successful = sum(1 for r in rows if int(r["liked"]) and int(r["meeting_agree"]))

    return {
        "avg_user_score": round(mean(scores), 2) if scores else 0.0,
        "likes_count": float(likes),
        "meetings_agreed": float(meetings),
        "successful_matches_ratio": round((successful / len(rows)) * 100.0, 2) if rows else 0.0,
    }
def _derive_nlp_label(liked: int, meeting_agree: int, user_score: Optional[int]) -> str:
    # Консервативные метки для обучения:
    # положительная: нравится и (согласия встретиться ещё или оценка >= 4)
    # отрицательная: не нравится или оценка <= 2
    # нейтральная: все двусмысленные случаи
    if int(liked) == 1 and (int(meeting_agree) == 1 or (user_score is not None and int(user_score) >= 4)):
        return "positive"
    if int(liked) == 0 or (user_score is not None and int(user_score) <= 2):
        return "negative"
    return "neutral"


# Выгружает пары профилей для NLP-обучения в CSV.
def export_nlp_dataset_csv(db: Database, output_path: str) -> str:
    _ensure_parent_dir(output_path)

    rows = db.get_nlp_feedback_dataset_rows()
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(NLP_EXPORT_FIELDS)

        for r in rows:
            liked = int(r["liked"])
            meeting_agree = int(r["meeting_agree"])
            user_score = int(r["user_score"]) if r["user_score"] is not None else None
            label = _derive_nlp_label(liked, meeting_agree, user_score)

            writer.writerow(
                [
                    r["id"],
                    r["from_user_id"],
                    r["to_user_id"],
                    r["from_about"] or "",
                    r["to_about"] or "",
                    json.dumps(r["from_answers"], ensure_ascii=False, sort_keys=True),
                    json.dumps(r["to_answers"], ensure_ascii=False, sort_keys=True),
                    liked,
                    meeting_agree,
                    user_score,
                    label,
                    r["created_at"],
                ]
            )
    return output_path




