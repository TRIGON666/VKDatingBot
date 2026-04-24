from __future__ import annotations

import csv
import json
import os
import random
import re
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional

from src.database import Database


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


def export_feedback_csv(db: Database, output_path: str) -> str:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    rows = db.get_feedback_rows()
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "from_user_id", "to_user_id", "liked", "meeting_agree", "user_score", "created_at"])
        for r in rows:
            writer.writerow(
                [
                    r["id"],
                    r["from_user_id"],
                    r["to_user_id"],
                    r["liked"],
                    r["meeting_agree"],
                    r["user_score"],
                    r["created_at"],
                ]
            )
    return output_path


def _derive_nlp_label(liked: int, meeting_agree: int, user_score: Optional[int]) -> str:
    # Консервативные метки для обучения:
    # положительная: нравится и (согласия встретиться ещё или оценка >= 4)
    # отрицательная: не нравится или оценка <= 2
    # йнейтральная: все двусмысленные случаи
    if int(liked) == 1 and (int(meeting_agree) == 1 or (user_score is not None and int(user_score) >= 4)):
        return "positive"
    if int(liked) == 0 or (user_score is not None and int(user_score) <= 2):
        return "negative"
    return "neutral"


def export_nlp_dataset_csv(db: Database, output_path: str) -> str:
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    rows = db.get_nlp_feedback_dataset_rows()
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
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
        )

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


_BAD_WORD_PATTERNS = [
    r"\bх+у+й+\w*\b",
    r"\bху[аеиоуыэюяйеё]\w*\b",
    r"\bпизд\w*\b",
    r"\bеб\w*\b",
    r"\bбля\w*\b",
    r"\bчм[оа]\w*\b",
]

URL_RE = re.compile(r"https?://\S+")
NON_ALNUM_RE = re.compile(r"[^a-zа-я0-9\s]", flags=re.IGNORECASE)
MULTI_SPACE_RE = re.compile(r"\s+")
BAD_WORD_RES = [re.compile(pattern, flags=re.IGNORECASE) for pattern in _BAD_WORD_PATTERNS]
BADWORD_TOKEN_SET = {"badword"}


def _sanitize_text(text: str) -> str:
    value = (text or "").lower().replace("ё", "е")
    value = URL_RE.sub(" ", value)
    value = NON_ALNUM_RE.sub(" ", value)
    value = MULTI_SPACE_RE.sub(" ", value).strip()
    for pattern in BAD_WORD_RES:
        value = pattern.sub(" badword ", value)
    value = MULTI_SPACE_RE.sub(" ", value).strip()
    return value


def _parse_answers(raw: str) -> Dict[str, Any]:
    try:
        loaded = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _valid_answers(answers: Dict[str, Any]) -> bool:
    required = {"age", "gender", "search_gender", "city"}
    if not required.issubset(set(answers.keys())):
        return False
    try:
        age = int(str(answers.get("age", "")).strip())
    except ValueError:
        return False
    return 12 <= age <= 99


def _is_low_signal(text: str) -> bool:
    tokens = text.split()
    if len(tokens) < 2:
        return True
    meaningful = [t for t in tokens if t not in BADWORD_TOKEN_SET]
    return len(meaningful) < 2


def _to_train_row(raw: Dict[str, str], idx: int) -> Optional[Dict[str, Any]]:
    left = _sanitize_text(raw.get("from_about", ""))
    right = _sanitize_text(raw.get("to_about", ""))
    if _is_low_signal(left) or _is_low_signal(right):
        return None

    label = (raw.get("label", "").strip().lower() or "neutral")
    if label not in {"positive", "neutral", "negative"}:
        return None

    from_answers = _parse_answers(raw.get("from_answers_json", "{}"))
    to_answers = _parse_answers(raw.get("to_answers_json", "{}"))
    if not _valid_answers(from_answers) or not _valid_answers(to_answers):
        return None

    from_age = int(str(from_answers.get("age", "0")).strip())
    to_age = int(str(to_answers.get("age", "0")).strip())

    return {
        "sample_id": f"s_{idx}",
        "feedback_id": raw.get("feedback_id", ""),
        "from_user_id": raw.get("from_user_id", ""),
        "to_user_id": raw.get("to_user_id", ""),
        "text_left": left,
        "text_right": right,
        "label": label,
        "from_city": "",
        "to_city": "",
        "age_diff": abs(from_age - to_age),
        "created_at": raw.get("created_at", ""),
    }


def _augment_row(row: Dict[str, Any], idx: int) -> Dict[str, Any]:
    augmented = dict(row)
    augmented["sample_id"] = f"aug_{idx}"
    # Остальныен аугментация для таскв творского текста.
    augmented["text_left"], augmented["text_right"] = row["text_right"], row["text_left"]
    augmented["from_city"], augmented["to_city"] = row["to_city"], row["from_city"]
    return augmented


def _build_synthetic_negative(rows: List[Dict[str, Any]], idx: int) -> Optional[Dict[str, Any]]:
    if len(rows) < 2:
        return None
    left = random.choice(rows)
    right = random.choice(rows)
    guard = 0
    while (left["from_user_id"] == right["to_user_id"] or left["text_left"] == right["text_right"]) and guard < 10:
        right = random.choice(rows)
        guard += 1

    if left["text_left"] == right["text_right"]:
        return None

    return {
        "sample_id": f"synneg_{idx}",
        "feedback_id": "synthetic",
        "from_user_id": left["from_user_id"],
        "to_user_id": right["to_user_id"],
        "text_left": left["text_left"],
        "text_right": right["text_right"],
        "label": "negative",
        "from_city": left["from_city"],
        "to_city": right["to_city"],
        "age_diff": abs(int(left["age_diff"]) - int(right["age_diff"])),
        "created_at": left["created_at"],
    }


def _build_synthetic_neutral(rows: List[Dict[str, Any]], idx: int) -> Optional[Dict[str, Any]]:
    if len(rows) < 2:
        return None
    left = random.choice(rows)
    right = random.choice(rows)
    if left["text_left"] == right["text_right"]:
        return None

    return {
        "sample_id": f"synneu_{idx}",
        "feedback_id": "synthetic",
        "from_user_id": left["from_user_id"],
        "to_user_id": right["to_user_id"],
        "text_left": left["text_left"],
        "text_right": right["text_right"],
        "label": "neutral",
        "from_city": left["from_city"],
        "to_city": right["to_city"],
        "age_diff": min(30, abs(int(left["age_diff"]) + int(right["age_diff"])) // 2),
        "created_at": left["created_at"],
    }


def build_nlp_training_dataset(
    input_csv_path: str,
    output_csv_path: str,
    include_neutral: bool = True,
    target_per_label: Optional[int] = None,
    random_seed: int = 42,
) -> Dict[str, int]:
    random.seed(random_seed)

    with open(input_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    converted: List[Dict[str, Any]] = []
    for idx, raw in enumerate(raw_rows, start=1):
        row = _to_train_row(raw, idx)
        if row is None:
            continue
        if not include_neutral and row["label"] == "neutral":
            continue
        converted.append(row)

    # Элиминирование точных текстовых пар с тем же меткой.
    unique: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for row in converted:
        key = (row["text_left"], row["text_right"], row["label"])
        unique[key] = row
    rows = list(unique.values())

    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    # Если осталась одна класс, синтезируйте базовые отрицания для виабильности обучения.
    if "negative" not in by_label and rows:
        synthetic_negatives: List[Dict[str, Any]] = []
        desired = max(1, len(rows) // 2)
        syn_idx = 1
        while len(synthetic_negatives) < desired:
            candidate = _build_synthetic_negative(rows, syn_idx)
            syn_idx += 1
            if candidate is None:
                break
            synthetic_negatives.append(candidate)
        if synthetic_negatives:
            by_label["negative"].extend(synthetic_negatives)
            rows.extend(synthetic_negatives)

    # Опционально синтезируйте нейтральный класс.
    if include_neutral and "neutral" not in by_label and rows:
        synthetic_neutral: List[Dict[str, Any]] = []
        desired = max(1, len(rows) // 3)
        syn_idx = 1
        while len(synthetic_neutral) < desired:
            candidate = _build_synthetic_neutral(rows, syn_idx)
            syn_idx += 1
            if candidate is None:
                break
            synthetic_neutral.append(candidate)
        if synthetic_neutral:
            by_label["neutral"].extend(synthetic_neutral)
            rows.extend(synthetic_neutral)

    if not by_label:
        parent = os.path.dirname(output_csv_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "sample_id",
                "feedback_id",
                "from_user_id",
                "to_user_id",
                "text_left",
                "text_right",
                "label",
                "from_city",
                "to_city",
                "age_diff",
                "created_at",
            ])
        return {"raw_rows": len(raw_rows), "clean_rows": 0, "final_rows": 0}

    target = target_per_label if target_per_label is not None else max(len(v) for v in by_label.values())
    final_rows: List[Dict[str, Any]] = []
    aug_idx = 1
    for label, group in by_label.items():
        final_rows.extend(group)
        while len(group) < target:
            if label == "negative":
                generated = _build_synthetic_negative(rows, aug_idx)
                if generated is None:
                    base = random.choice(group)
                    generated = _augment_row(base, aug_idx)
                else:
                    generated["label"] = "negative"
            elif label == "neutral":
                generated = _build_synthetic_neutral(rows, aug_idx)
                if generated is None:
                    base = random.choice(group)
                    generated = _augment_row(base, aug_idx)
                    generated["label"] = "neutral"
                else:
                    generated["label"] = "neutral"
            else:
                base = random.choice(group)
                generated = _augment_row(base, aug_idx)
                generated["label"] = "positive"

            group.append(generated)
            final_rows.append(group[-1])
            aug_idx += 1

    random.shuffle(final_rows)

    parent = os.path.dirname(output_csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(output_csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "feedback_id",
                "from_user_id",
                "to_user_id",
                "text_left",
                "text_right",
                "label",
                "from_city",
                "to_city",
                "age_diff",
                "created_at",
            ]
        )
        for row in final_rows:
            writer.writerow(
                [
                    row["sample_id"],
                    row["feedback_id"],
                    row["from_user_id"],
                    row["to_user_id"],
                    row["text_left"],
                    row["text_right"],
                    row["label"],
                    row["from_city"],
                    row["to_city"],
                    row["age_diff"],
                    row["created_at"],
                ]
            )

    stats = {
        "raw_rows": len(raw_rows),
        "clean_rows": len(rows),
        "final_rows": len(final_rows),
    }
    for label in sorted(by_label.keys()):
        stats[f"label_{label}"] = sum(1 for r in final_rows if r["label"] == label)
    return stats
