from __future__ import annotations

import csv
import io
import json
import logging
import os
from contextlib import redirect_stderr
from functools import lru_cache
from importlib import metadata
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

LABEL_TO_ID = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}
logger = logging.getLogger(__name__)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

LEFT_PREFIX = "left_profile: "
PAIR_SEPARATOR = " || right_profile: "

TOPIC_GROUPS = {
    "active": {"спорт", "трениров", "поход", "путешеств", "бег", "фитнес", "движен", "гор", "велосипед"},
    "home": {"дом", "книг", "чай", "сериал", "настол", "уют", "тишин", "спокойн"},
    "social": {"вечерин", "клуб", "бар", "танц", "компан", "общен", "знакомств", "друз"},
    "family": {"сем", "отношен", "будущ", "стабильн", "довер", "брак", "дет"},
    "career": {"карьер", "работ", "бизнес", "проект", "цель", "развит", "деньг"},
    "creative": {"музык", "кино", "искусств", "выстав", "концерт", "творч", "дизайн"},
    "travel": {"путешеств", "поезд", "город", "стран", "дорог", "удален", "свобод"},
}
CASUAL_MARKERS = {"легк", "свобод", "обязательств", "флирт"}
CONFLICTING_TOPIC_PAIRS = {
    ("home", "social"),
    ("home", "active"),
    ("family", "casual"),
    ("career", "home"),
}

_SENTENCE_MODEL = None
_SENTENCE_MODEL_NAME = None
_SENTENCE_MODEL_UNAVAILABLE_NAMES: set[str] = set()


# Создает родительскую папку для будущего файла.
def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


# Достает macro precision, recall и F1 из отчета обучения.
def extract_macro_metrics(report_payload: dict) -> Tuple[float, float, float]:
    report = report_payload.get("classification_report", {})
    macro_avg = report.get("macro avg", {}) if isinstance(report, dict) else {}
    precision = float(macro_avg.get("precision", 0.0))
    recall = float(macro_avg.get("recall", 0.0))
    f1 = float(macro_avg.get("f1-score", report_payload.get("macro_f1", 0.0)))
    return precision, recall, f1
@lru_cache(maxsize=2)
def _load_model_payload(model_path: str) -> Dict[str, object]:
    if not os.path.exists(model_path):
        raise RuntimeError(f"Модель не найдена: {model_path}")
    return joblib.load(model_path)
def _safe_div(left: float, right: float) -> float:
    return left / right if right else 0.0
def _version_at_least(version: str, minimum: Tuple[int, ...]) -> bool:
    parts = []
    for chunk in version.replace("-", ".").split("."):
        if not chunk.isdigit():
            break
        parts.append(int(chunk))
    return tuple(parts) >= minimum
def _can_load_sentence_transformers() -> bool:
    """Skip noisy imports when the local torch stack is known to be incompatible."""
    if os.getenv("NLP_DISABLE_PRETRAINED", "").lower() in {"1", "true", "yes", "on"}:
        return False
    try:
        torch_version = metadata.version("torch")
    except metadata.PackageNotFoundError:
        return False
    return _version_at_least(torch_version, (2, 4))
def _get_sentence_model(model_name: Optional[str] = None):
    """Load a pretrained sentence-transformers model lazily.

    If the dependency/model is unavailable, return None and let the local
    TF-IDF/heuristic pipeline keep working.
    """
    global _SENTENCE_MODEL, _SENTENCE_MODEL_NAME
    if model_name is None:
        try:
            from src.config import NLP_PRETRAINED_MODEL_NAME

            model_name = NLP_PRETRAINED_MODEL_NAME
        except Exception:
            model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    if _SENTENCE_MODEL is not None and _SENTENCE_MODEL_NAME == model_name:
        return _SENTENCE_MODEL
    if model_name in _SENTENCE_MODEL_UNAVAILABLE_NAMES:
        return None
    if not _can_load_sentence_transformers():
        _SENTENCE_MODEL_UNAVAILABLE_NAMES.add(model_name)
        return None

    try:
        with redirect_stderr(io.StringIO()):
            from sentence_transformers import SentenceTransformer

        _SENTENCE_MODEL = SentenceTransformer(model_name, local_files_only=True)
        _SENTENCE_MODEL_NAME = model_name
        return _SENTENCE_MODEL
    except Exception:
        try:
            with redirect_stderr(io.StringIO()):
                from sentence_transformers import SentenceTransformer
            _SENTENCE_MODEL = SentenceTransformer(model_name)
            _SENTENCE_MODEL_NAME = model_name
            return _SENTENCE_MODEL
        except Exception:
            _SENTENCE_MODEL_UNAVAILABLE_NAMES.add(model_name)
            logger.debug("Pretrained sentence model is unavailable: %s", model_name)
            return None
def _cosine_from_vectors(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


# Считает семантическую похожесть через pretrained-модель, если она доступна.
def pretrained_semantic_score(text_left: str, text_right: str, model_name: Optional[str] = None) -> Optional[float]:
    model = _get_sentence_model(model_name)
    if model is None:
        return None
    try:
        try:
            embeddings = model.encode(
                [text_left or "", text_right or ""],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except TypeError:
            embeddings = model.encode([text_left or "", text_right or ""], normalize_embeddings=True)
        cosine = _cosine_from_vectors(np.asarray(embeddings[0]), np.asarray(embeddings[1]))
        # Convert cosine [-1, 1] to a conservative 0..100 compatibility range.
        return round(max(0.0, min(100.0, (cosine + 1.0) * 50.0)), 2)
    except Exception:
        logger.warning("Failed to calculate pretrained semantic score", exc_info=True)
        return None
def _split_pair_text(pair_text: str) -> Tuple[str, str]:
    value = pair_text or ""
    if value.startswith(LEFT_PREFIX):
        value = value[len(LEFT_PREFIX):]
    if PAIR_SEPARATOR in value:
        left, right = value.split(PAIR_SEPARATOR, 1)
        return left.strip(), right.strip()
    return value.strip(), ""


# Обрабатывает текстовые токены.
def _tokens(text: str) -> List[str]:
    try:
        from src.nlp_preprocessing import preprocess_profile

        prepared = preprocess_profile(text or "")
    except Exception:
        logger.debug("Text preprocessing failed, using simple fallback", exc_info=True)
        prepared = (text or "").lower()
    return [token for token in prepared.split() if len(token) >= 3]


# Считает тематические признаки текста.
def _topic_vector(tokens: Iterable[str]) -> Dict[str, float]:
    token_list = list(tokens)
    result: Dict[str, float] = {}
    for name, markers in TOPIC_GROUPS.items():
        hits = sum(1 for token in token_list if any(marker in token for marker in markers))
        result[name] = min(1.0, hits / 3.0)
    result["casual"] = min(1.0, sum(1 for token in token_list if any(marker in token for marker in CASUAL_MARKERS)) / 2.0)
    return result
def _pair_heuristic_score(text_left: str, text_right: str) -> float:
    left_tokens = _tokens(text_left)
    right_tokens = _tokens(text_right)
    left_set = set(left_tokens)
    right_set = set(right_tokens)
    if not left_set or not right_set:
        return 50.0

    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    jaccard = _safe_div(intersection, union)
    containment = max(_safe_div(intersection, len(left_set)), _safe_div(intersection, len(right_set)))

    left_topics = _topic_vector(left_tokens)
    right_topics = _topic_vector(right_tokens)
    topic_names = set(left_topics) | set(right_topics)
    topic_similarity = 1.0 - (
        sum(abs(left_topics.get(name, 0.0) - right_topics.get(name, 0.0)) for name in topic_names)
        / max(1, len(topic_names))
    )

    conflict_penalty = 0.0
    for left_topic, right_topic in CONFLICTING_TOPIC_PAIRS:
        conflict_penalty = max(
            conflict_penalty,
            left_topics.get(left_topic, 0.0) * right_topics.get(right_topic, 0.0),
            right_topics.get(left_topic, 0.0) * left_topics.get(right_topic, 0.0),
        )

    score = (jaccard * 22.0) + (containment * 18.0) + (topic_similarity * 60.0) - (conflict_penalty * 42.0)
    return max(0.0, min(100.0, score))


class PairCompatibilityFeatures(BaseEstimator, TransformerMixin):
    """Additional pair-level features for the pretrained compatibility model."""

    # Поддерживает интерфейс sklearn и возвращает текущий трансформер.
    def fit(self, X: Iterable[str], y: Optional[Iterable[int]] = None) -> "PairCompatibilityFeatures":
        return self

    # Преобразует входные данные в признаки для модели.
    def transform(self, X: Iterable[str]) -> np.ndarray:
        rows: List[List[float]] = []
        for pair_text in X:
            left, right = _split_pair_text(str(pair_text))
            left_tokens = _tokens(left)
            right_tokens = _tokens(right)
            left_set = set(left_tokens)
            right_set = set(right_tokens)
            intersection = len(left_set & right_set)
            jaccard = _safe_div(intersection, len(left_set | right_set))
            dice = _safe_div(2 * intersection, len(left_set) + len(right_set))
            containment = max(_safe_div(intersection, len(left_set)), _safe_div(intersection, len(right_set)))
            len_ratio = min(_safe_div(len(left_tokens), len(right_tokens)), _safe_div(len(right_tokens), len(left_tokens)))

            left_topics = _topic_vector(left_tokens)
            right_topics = _topic_vector(right_tokens)
            topic_names = sorted(set(left_topics) | set(right_topics))
            topic_similarity = 1.0 - (
                sum(abs(left_topics.get(name, 0.0) - right_topics.get(name, 0.0)) for name in topic_names)
                / max(1, len(topic_names))
            )
            conflict = 0.0
            for left_topic, right_topic in CONFLICTING_TOPIC_PAIRS:
                conflict = max(
                    conflict,
                    left_topics.get(left_topic, 0.0) * right_topics.get(right_topic, 0.0),
                    right_topics.get(left_topic, 0.0) * left_topics.get(right_topic, 0.0),
                )

            rows.append(
                [
                    jaccard,
                    dice,
                    containment,
                    len_ratio,
                    topic_similarity,
                    conflict,
                    _pair_heuristic_score(left, right) / 100.0,
                    min(1.0, float(len(left_tokens)) / 80.0),
                    min(1.0, float(len(right_tokens)) / 80.0),
                ]
            )
        return np.asarray(rows, dtype=float)


class PretrainedEmbeddingFeatures(BaseEstimator, TransformerMixin):
    """Compact features from a pretrained sentence-transformers model.

    The model itself is not fine-tuned here. The project trains a lightweight
    compatibility classifier on top of frozen pretrained embeddings, which is
    fast and works with synthetic plus collected user data.
    """

    # Инициализирует объект и сохраняет нужные зависимости.
    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name

    # Поддерживает интерфейс sklearn и возвращает текущий трансформер.
    def fit(self, X: Iterable[str], y: Optional[Iterable[int]] = None) -> "PretrainedEmbeddingFeatures":
        return self

    # Преобразует входные данные в признаки для модели.
    def transform(self, X: Iterable[str]) -> np.ndarray:
        pairs = [_split_pair_text(str(value)) for value in X]
        model = _get_sentence_model(self.model_name)
        if model is None:
            return np.zeros((len(pairs), 6), dtype=float)

        left_texts = [left for left, _ in pairs]
        right_texts = [right for _, right in pairs]
        try:
            left_embeddings = np.asarray(model.encode(left_texts, normalize_embeddings=True, show_progress_bar=False))
            right_embeddings = np.asarray(model.encode(right_texts, normalize_embeddings=True, show_progress_bar=False))
        except TypeError:
            left_embeddings = np.asarray(model.encode(left_texts, normalize_embeddings=True))
            right_embeddings = np.asarray(model.encode(right_texts, normalize_embeddings=True))
        except Exception:
            logger.warning("Failed to encode pretrained embedding features", exc_info=True)
            return np.zeros((len(pairs), 6), dtype=float)

        rows: List[List[float]] = []
        for left_vec, right_vec in zip(left_embeddings, right_embeddings):
            diff = np.abs(left_vec - right_vec)
            product = left_vec * right_vec
            cosine = _cosine_from_vectors(left_vec, right_vec)
            rows.append(
                [
                    cosine,
                    float(np.mean(diff)),
                    float(np.max(diff)),
                    float(np.mean(product)),
                    float(np.std(diff)),
                    float(np.linalg.norm(diff) / max(1.0, np.sqrt(diff.shape[0]))),
                ]
            )
        return np.asarray(rows, dtype=float)
def _pair_to_text(text_left: str, text_right: str) -> str:
    try:
        from src.nlp_preprocessing import preprocess_profile

        left = preprocess_profile(text_left or "")
        right = preprocess_profile(text_right or "")
    except Exception:
        logger.debug("Text preprocessing failed, using raw text", exc_info=True)
        left = (text_left or "").strip()
        right = (text_right or "").strip()
    return f"{LEFT_PREFIX}{left}{PAIR_SEPARATOR}{right}"


# Готовит или читает данные для обучения.
def _read_dataset(csv_path: str) -> Tuple[List[str], List[int]]:
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


# Готовит или читает данные для обучения.
def _read_many_datasets(csv_paths: Iterable[str]) -> Tuple[List[str], List[int], Dict[str, int]]:
    all_features: List[str] = []
    all_labels: List[int] = []
    stats: Dict[str, int] = {}
    for path in csv_paths:
        if not path or not os.path.exists(path):
            continue
        features, labels = _read_dataset(path)
        all_features.extend(features)
        all_labels.extend(labels)
        stats[path] = len(labels)
    return all_features, all_labels, stats


# Обучает модель текстовой совместимости и сохраняет отчет.
def train_text_compatibility_model(
    dataset_csv_path: str,
    model_output_path: str,
    report_output_path: str,
    validation_csv_path: Optional[str] = None,
    test_csv_path: Optional[str] = None,
    extra_dataset_paths: Optional[List[str]] = None,
    real_data_weight: int = 3,
    random_seed: int = 42,
) -> Dict[str, object]:
    X, y = _read_dataset(dataset_csv_path)
    extra_X, extra_y, extra_stats = _read_many_datasets(extra_dataset_paths or [])
    if extra_X:
        repeat = max(1, int(real_data_weight))
        X = X + (extra_X * repeat)
        y = y + (extra_y * repeat)

    if len(X) < 50:
        raise RuntimeError("Датасет слишком мал для обучения. Нужно минимум 50 строк.")

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
                    ngram_range=(1, 3),
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
                    ngram_range=(3, 6),
                    min_df=1,
                    max_df=0.995,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            ("pair_features", PairCompatibilityFeatures()),
            ("pretrained_embeddings", PretrainedEmbeddingFeatures()),
        ]
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=random_seed,
        C=1.5,
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
    try:
        from src.config import NLP_PRETRAINED_MODEL_NAME
    except Exception:
        NLP_PRETRAINED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
            "label_to_id": LABEL_TO_ID,
            "model_kind": "pretrained_pair_logreg_v2",
            "score_blend": {"model_positive": 0.48, "pair_heuristic": 0.32, "pretrained_semantic": 0.20},
            "pretrained_model_name": NLP_PRETRAINED_MODEL_NAME,
        },
        model_output_path,
    )
    _load_model_payload.cache_clear()

    report_payload: Dict[str, object] = {
        "dataset": dataset_csv_path,
        "extra_datasets": extra_stats,
        "real_data_weight": real_data_weight,
        "validation_dataset": validation_csv_path,
        "train_size": len(X_train),
        "validation_size": len(X_test),
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "model_kind": "pretrained_pair_logreg_v2",
    }
    if test_report:
        report_payload.update(test_report)

    ensure_parent_dir(report_output_path)
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, ensure_ascii=False, indent=2)

    return report_payload


# Возвращает прогноз совместимости для пары текстов.
def predict_text_compatibility(text_left: str, text_right: str, model_path: str) -> Dict[str, object]:
    payload = _load_model_payload(model_path)
    vectorizer = payload["vectorizer"]
    model: LogisticRegression = payload["model"]

    X = vectorizer.transform([_pair_to_text(text_left, text_right)])
    pred_id = int(model.predict(X)[0])
    model_label = ID_TO_LABEL.get(pred_id, "neutral")
    probs = model.predict_proba(X)[0]
    model_positive_score = float(probs[LABEL_TO_ID["positive"]] * 100.0)
    heuristic_score = _pair_heuristic_score(text_left, text_right)

    pretrained_score = pretrained_semantic_score(text_left, text_right)

    blend = payload.get("score_blend", {}) if isinstance(payload, dict) else {}
    model_weight = float(blend.get("model_positive", 0.72))
    heuristic_weight = float(blend.get("pair_heuristic", 0.28))
    pretrained_weight = float(blend.get("pretrained_semantic", 0.0)) if pretrained_score is not None else 0.0
    weight_sum = model_weight + heuristic_weight + pretrained_weight
    score = model_positive_score if weight_sum <= 0 else (
        (model_positive_score * model_weight)
        + (heuristic_score * heuristic_weight)
        + ((pretrained_score or 0.0) * pretrained_weight)
    ) / weight_sum

    if heuristic_score <= 25.0:
        score = min(score, heuristic_score + 25.0)
    elif heuristic_score >= 60.0:
        score = max(score, heuristic_score)

    if score >= 60:
        label = "positive"
    elif score <= 45:
        label = "negative"
    else:
        label = "neutral"

    return {
        "label": label,
        "model_label": model_label,
        "score_percent": round(score, 2),
        "model_score_percent": round(model_positive_score, 2),
        "heuristic_score_percent": round(heuristic_score, 2),
        "pretrained_semantic_score_percent": round(pretrained_score, 2) if pretrained_score is not None else None,
        "proba_negative": round(float(probs[LABEL_TO_ID["negative"]]), 4),
        "proba_neutral": round(float(probs[LABEL_TO_ID["neutral"]]), 4),
        "proba_positive": round(float(probs[LABEL_TO_ID["positive"]]), 4),
    }
