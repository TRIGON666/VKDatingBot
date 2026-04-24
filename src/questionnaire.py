from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Question:
    """Один вопрос анкеты с весом для итогового скоринга."""
    key: str
    text: str
    options: List[str]
    weight: float


QUESTIONS: List[Question] = [
    Question("activity", "Образ жизни", ["домашний", "смешанный", "активный"], 1.2),
    Question("communication", "Стиль общения", ["спокойный", "нейтральный", "эмоциональный"], 1.0),
    Question("values", "Приоритеты", ["карьера", "семья", "баланс"], 1.4),
    Question("tempo", "Темп жизни", ["медленный", "средний", "быстрый"], 0.8),
]


def calculate_questionnaire_compatibility(
    left_answers: Dict[str, str], right_answers: Dict[str, str]
) -> float:
    """Считает совместимость анкет по совпавшим ответам с учетом весов."""
    weighted_matches = 0.0
    total_weight = 0.0

    for q in QUESTIONS:
        total_weight += q.weight
        if left_answers.get(q.key) == right_answers.get(q.key):
            weighted_matches += q.weight

    if total_weight == 0:
        return 0.0
    # Формула: (сумма весов совпадений / сумма всех весов) * 100.
    score = (weighted_matches / total_weight) * 100.0
    return round(score, 2)
