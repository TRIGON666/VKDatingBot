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


# Считает совпадения короткой анкеты с учетом весов.
def calculate_questionnaire_compatibility(left_answers: Dict[str, str], right_answers: Dict[str, str]) -> float:
    """Считает совместимость анкет по совпавшим ответам с учетом весов."""
    total_weight = sum(q.weight for q in QUESTIONS)
    if total_weight == 0:
        return 0.0
    matched_weight = sum(q.weight for q in QUESTIONS if left_answers.get(q.key) == right_answers.get(q.key))
    return round((matched_weight / total_weight) * 100.0, 2)
