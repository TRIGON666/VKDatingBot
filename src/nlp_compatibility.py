from __future__ import annotations

import csv
import json
import logging
import os
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
logger = logging.getLogger(__name__)


def ensure_parent_dir(path: str) -> None:
    """Создает родительскую директорию для файла, если она отсутствует."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def extract_macro_metrics(report_payload: dict) -> Tuple[float, float, float]:
    """Извлекает Precision/Recall/F1 из блока macro avg отчета sklearn."""
    classification_report = report_payload.get("classification_report", {})
    macro_avg = classification_report.get("macro avg", {}) if isinstance(classification_report, dict) else {}
    precision = float(macro_avg.get("precision", 0.0))
    recall = float(macro_avg.get("recall", 0.0))
    f1 = float(macro_avg.get("f1-score", report_payload.get("macro_f1", 0.0)))
    return precision, recall, f1


@lru_cache(maxsize=2)
def _load_model_payload(model_path: str) -> Dict[str, object]:
    """Загружает и кэширует модельный payload с диска."""
    if not os.path.exists(model_path):
        raise RuntimeError(f"Модель не найдена: {model_path}")
    return joblib.load(model_path)


def _pair_to_text(text_left: str, text_right: str) -> str:
    """Преобразует пару профилей в единую строку признаков."""
    try:
        from src.nlp_preprocessing import preprocess_profile
        left = preprocess_profile(text_left or "")
        right = preprocess_profile(text_right or "")
    except Exception:
        logger.debug("Не удалось предобработать текст профиля, используется исходный текст", exc_info=True)
        left = (text_left or "").strip()
        right = (text_right or "").strip()
    
    return f"left_profile: {left} || right_profile: {right}"


def _read_dataset(csv_path: str) -> Tuple[List[str], List[int]]:
    """Читает датасет и возвращает список текстовых признаков и меток."""
    features: List[str] = []
    labels: List[int] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = str(row.get("label", "")).strip().lower()
            if label not in LABEL_TO_ID:
                continue
            text_left = str(row.get("text_left") or row.get("from_about") or "")
            text_right = str(row.get("text_right") or row.get("to_about") or "")
            if not text_left.strip() or not text_right.strip():
                continue
            features.append(_pair_to_text(text_left, text_right))
            labels.append(LABEL_TO_ID[label])
    return features, labels


def train_text_compatibility_model(
    dataset_csv_path: str,
    model_output_path: str,
    report_output_path: str,
    validation_csv_path: Optional[str] = None,
    test_csv_path: Optional[str] = None,
    random_seed: int = 42,
) -> Dict[str, object]:
    """Обучает модель совместимости и сохраняет модель+отчет."""
    X, y = _read_dataset(dataset_csv_path)
    if len(X) < 50:
        raise RuntimeError("Датасет слишком мал для надежного обучения. Нужно минимум 50 строк.")

    if validation_csv_path:
        X_train, y_train = X, y
        X_test, y_test = _read_dataset(validation_csv_path)
        if len(X_test) < 10:
            raise RuntimeError("Валидационный датасет слишком мал для надежной оценки.")
    else:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=random_seed,
                stratify=y,
            )
        except ValueError:
            # Запасной режим для малых/несбалансированных выборок.
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=random_seed,
                stratify=None,
            )

    vectorizer = FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.995,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    max_df=0.995,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
        ]
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=random_seed,
        C=4.0,
    )
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    acc = float(accuracy_score(y_test, preds))
    macro_f1 = float(f1_score(y_test, preds, average="macro"))

    labels_sorted = [LABEL_TO_ID["negative"], LABEL_TO_ID["neutral"], LABEL_TO_ID["positive"]]
    report = classification_report(
        y_test,
        preds,
        labels=labels_sorted,
        target_names=["negative", "neutral", "positive"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, preds, labels=labels_sorted)

    test_report = None
    if test_csv_path:
        X_final, y_final = _read_dataset(test_csv_path)
        if len(X_final) < 10:
            raise RuntimeError("Тестовый датасет слишком мал для финальной оценки.")
        X_final_vec = vectorizer.transform(X_final)
        final_preds = model.predict(X_final_vec)
        final_acc = float(accuracy_score(y_final, final_preds))
        final_f1 = float(f1_score(y_final, final_preds, average="macro"))
        test_report = {
            "final_test_dataset": test_csv_path,
            "final_test_size": len(X_final),
            "final_test_accuracy": round(final_acc, 4),
            "final_test_macro_f1": round(final_f1, 4),
        }

    ensure_parent_dir(model_output_path)
    joblib.dump({"vectorizer": vectorizer, "model": model, "label_to_id": LABEL_TO_ID}, model_output_path)
    _load_model_payload.cache_clear()

    report_payload: Dict[str, object] = {
        "dataset": dataset_csv_path,
        "validation_dataset": validation_csv_path,
        "train_size": len(X_train),
        "validation_size": len(X_test),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    if test_report:
        report_payload.update(test_report)

    ensure_parent_dir(report_output_path)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)

    return report_payload


def predict_text_compatibility(text_left: str, text_right: str, model_path: str) -> Dict[str, object]:
    """Возвращает предсказание совместимости и вероятности классов."""
    payload = _load_model_payload(model_path)
    vectorizer = payload["vectorizer"]
    model: LogisticRegression = payload["model"]

    X = vectorizer.transform([_pair_to_text(text_left, text_right)])
    pred_id = int(model.predict(X)[0])
    probs = model.predict_proba(X)[0]
    positive_score = float(probs[LABEL_TO_ID["positive"]] * 100.0)

    return {
        "label": ID_TO_LABEL.get(pred_id, "neutral"),
        "score_percent": round(positive_score, 2),
        "proba_negative": round(float(probs[LABEL_TO_ID["negative"]]), 4),
        "proba_neutral": round(float(probs[LABEL_TO_ID["neutral"]]), 4),
        "proba_positive": round(float(probs[LABEL_TO_ID["positive"]]), 4),
    }
