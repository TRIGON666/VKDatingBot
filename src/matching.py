from __future__ import annotations

import os
import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.config import NLP_MODEL_PATH
from src.database import UserProfile
from src.nlp_compatibility import predict_text_compatibility
from src.psychology_questions import get_compatibility_recommendation
from src.questionnaire import calculate_questionnaire_compatibility
from src.text_analysis import tfidf_compatibility


logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    user_id: int
    questionnaire_score: float
    tfidf_score: float
    nlp_score: Optional[float]
    psychology_score: Optional[float]
    combined_score: float
def _resolve_model_path() -> str:
    """Возвращает путь к модели с приоритетом env -> config."""
    return os.getenv("NLP_MODEL_PATH", "").strip() or NLP_MODEL_PATH
def _nlp_text_score(base_about: str, candidate_about: str, model_path: str) -> float | None:
    """Считает NLP-скор пары текстов, либо None при недоступной модели."""
    try:
        payload = predict_text_compatibility(base_about, candidate_about, model_path)
        return float(payload.get("score_percent", 0.0))
    except (FileNotFoundError, RuntimeError, KeyError, ValueError, TypeError):
        # Нет валидного прогноза — этот компонент не учитываем.
        return None
    except Exception as e:
        # Логирование от неожиданных ошибок.
        logger.warning("Unexpected error in NLP scoring: %s", e)
        return None
def _calibrate_nlp_score(raw_score: float) -> float:
    """Сглаживает крайние NLP-оценки к центру шкалы."""
    # Формула калибровки: 50 + (raw - 50) * 0.55.
    centered = 50.0 + ((raw_score - 50.0) * 0.55)
    return max(0.0, min(100.0, round(centered, 2)))
def _psychology_compatibility_score(
    left_scores: Optional[Dict[str, float]],
    right_scores: Optional[Dict[str, float]],
) -> Optional[float]:
    """Возвращает психологическую совместимость, если есть данные у обеих сторон."""
    if not left_scores or not right_scores:
        return None
    payload = get_compatibility_recommendation(left_scores, right_scores)
    value = payload.get("overall_score")
    if value is None:
        return None
    return float(value)


# Ранжирует кандидатов по доступным компонентам совместимости.
def rank_matches(
    base: UserProfile,
    candidates: List[UserProfile],
    psychology_scores_by_user: Optional[Dict[int, Dict[str, float]]] = None,
) -> List[MatchResult]:
    """Ранжирует кандидатов по комбинированной совместимости."""
    psychology_scores_by_user = psychology_scores_by_user or {}
    base_psych = psychology_scores_by_user.get(base.user_id)
    model_path = _resolve_model_path()

    results: List[MatchResult] = []
    for cand in candidates:
        questionnaire_score = calculate_questionnaire_compatibility(base.questionnaire, cand.questionnaire)
        tfidf_score = tfidf_compatibility(base.about_text, cand.about_text)
        raw_nlp_score = _nlp_text_score(base.about_text, cand.about_text, model_path)
        nlp_score = _calibrate_nlp_score(raw_nlp_score) if raw_nlp_score is not None else None

        psychology_score = _psychology_compatibility_score(base_psych, psychology_scores_by_user.get(cand.user_id))

        # Итог: взвешенное среднее доступных компонент.
        components: List[tuple[float, float]] = [
            (questionnaire_score, 0.35),
            (tfidf_score, 0.15),
        ]
        if nlp_score is not None:
            components.append((nlp_score, 0.30))
        if psychology_score is not None:
            components.append((psychology_score, 0.20))

        weight_sum = sum(w for _, w in components)
        combined_score = round(sum(score * w for score, w in components) / weight_sum, 2) if weight_sum else 0.0

        results.append(
            MatchResult(
                user_id=cand.user_id,
                questionnaire_score=questionnaire_score,
                tfidf_score=tfidf_score,
                nlp_score=nlp_score,
                psychology_score=psychology_score,
                combined_score=combined_score,
            )
        )

    # Keep relevance first, but randomize candidates inside close score bands so browsing
    # is less repetitive while still moving from strongest to weakest matches.
    sorted_results = sorted(results, key=lambda v: v.combined_score, reverse=True)
    bands: Dict[int, List[MatchResult]] = {}
    for item in sorted_results:
        band = int(item.combined_score // 10)
        bands.setdefault(band, []).append(item)

    randomized: List[MatchResult] = []
    rng = random.Random()
    for band in sorted(bands.keys(), reverse=True):
        group = bands[band]
        rng.shuffle(group)
        randomized.extend(group)
    return randomized
