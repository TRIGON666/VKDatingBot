from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Iterable, List, Set

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.constants import BASE_RU_STOPWORDS


RU_STOPWORDS_SET = set(BASE_RU_STOPWORDS)
NON_ALNUM_RE = re.compile(r"[^a-zа-яё0-9\s]", flags=re.IGNORECASE)

NORM_MAP = {
    "айти": "it",
    "ит": "it",
    "программист": "it",
    "разработчик": "it",
    "кодер": "it",
    "дев": "it",
    "девопс": "it",
    "тестировщик": "it",
    "дизайнер": "design",
    "путешествия": "путешествие",
    "путешествовать": "путешествие",
    "путешествую": "путешествие",
    "поездки": "путешествие",
    "поездка": "путешествие",
    "трип": "путешествие",
    "поход": "путешествие",
    "походы": "путешествие",
    "спортзал": "спорт",
    "зал": "спорт",
    "качалка": "спорт",
    "фитнес": "спорт",
    "бег": "спорт",
    "бегаю": "спорт",
    "йога": "спорт",
    "плавание": "спорт",
    "велосипед": "спорт",
    "тренировки": "тренировка",
    "тренируюсь": "тренировка",
    "киношка": "кино",
    "кинотеатр": "кино",
    "фильмы": "фильм",
    "сериалы": "сериал",
    "книги": "книга",
    "книжки": "книга",
    "чтение": "книга",
    "читать": "книга",
    "музон": "музыка",
    "слушаю": "музыка",
    "концерты": "концерт",
    "гулять": "прогулка",
    "гуляю": "прогулка",
    "парк": "прогулка",
    "интроверт": "спокойный",
    "экстраверт": "активный",
    "домосед": "домашний",
    "домоседка": "домашний",
    "семья": "семейный",
    "дети": "семейный",
    "отношения": "отношения",
    "серьезные": "серьезно",
    "брак": "серьезно",
    "флирт": "легко",
    "общение": "общение",
    "чат": "общение",
    "переписка": "общение",
    "дружба": "дружба",
    "настолки": "настольные_игры",
    "игры": "игра",
    "гейминг": "игра",
    "учеба": "учеба",
    "студент": "учеба",
    "работаю": "карьера",
    "работа": "карьера",
    "карьера": "карьера",
    "бизнес": "карьера",
    "кофейни": "кофе",
    "готовить": "кулинария",
    "готовка": "кулинария",
    "рестораны": "еда",
    "тусовки": "вечеринка",
    "вечеринки": "вечеринка",
    "клуб": "вечеринка",
    "клубы": "вечеринка",
    "природа": "природа",
    "горы": "природа",
    "море": "природа",
    "питомцы": "животные",
    "животные": "животные",
    "саморазвитие": "развитие",
    "юмор": "юмор",
    "мемы": "юмор",
}

RU_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "ого",
    "ему",
    "ому",
    "ее",
    "ие",
    "ые",
    "ой",
    "ий",
    "ый",
    "ая",
    "ое",
    "ей",
    "ам",
    "ям",
    "ах",
    "ях",
    "ию",
    "ью",
    "ия",
    "ья",
    "ов",
    "ев",
    "ом",
    "ем",
    "а",
    "я",
    "ы",
    "и",
    "е",
    "у",
    "ю",
)

TOPIC_GROUPS = {
    "active": {"спорт", "тренировка", "путешествие", "природа", "активный"},
    "home": {"домашний", "книга", "сериал", "чай", "настольные_игры", "спокойный"},
    "social": {"вечеринка", "общение", "дружба", "клуб", "бар"},
    "family": {"семейный", "отношения", "серьезно"},
    "career": {"карьера", "работа", "бизнес", "развитие"},
    "creative": {"музыка", "кино", "концерт", "design"},
}

CONFLICTING_TOPIC_PAIRS = {("home", "social"), ("home", "active"), ("family", "social")}
def _stem_ru(token: str) -> str:
    if token.isdigit() or len(token) < 4:
        return token
    for suffix in RU_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


# Обрабатывает текстовые токены.
def _normalize_token(token: str) -> str:
    mapped = NORM_MAP.get(token, token)
    return _stem_ru(mapped)


# Обрабатывает текстовые токены.
def _tokenize(text: str) -> List[str]:
    normalized = (text or "").lower().replace("ё", "е").strip()
    normalized = NON_ALNUM_RE.sub(" ", normalized)
    tokens: List[str] = []
    for raw_token in normalized.split():
        if len(raw_token) < 2 and not raw_token.isdigit():
            continue
        token = _normalize_token(raw_token)
        if token in RU_STOPWORDS_SET:
            continue
        tokens.append(token)
    return tokens


# Очищает текст для TF-IDF сравнения.
def preprocess_text(text: str) -> str:
    tokens = _tokenize(text)
    topics = sorted(_topic_set(tokens))
    return " ".join([*tokens, *topics, *topics])
def _safe_similarity(left_matrix, right_matrix) -> float:
    return float(cosine_similarity(left_matrix, right_matrix)[0][0])
def _weighted_jaccard_similarity(first_text: str, second_text: str) -> float:
    first_tokens = _tokenize(first_text)
    second_tokens = _tokenize(second_text)
    if not first_tokens or not second_tokens:
        return 0.0

    left = Counter(first_tokens)
    right = Counter(second_tokens)
    words = set(left) | set(right)
    intersection = sum(min(left[word], right[word]) for word in words)
    union = sum(max(left[word], right[word]) for word in words)
    return float(intersection / union) if union else 0.0


# Считает тематические признаки текста.
def _topic_set(tokens: Iterable[str]) -> Set[str]:
    token_set = set(tokens)
    topics = set()
    for topic, markers in TOPIC_GROUPS.items():
        if any(marker in token or token in marker for token in token_set for marker in markers):
            topics.add(topic)
    return topics


# Считает тематические признаки текста.
def _topic_similarity(first_text: str, second_text: str) -> float:
    first_topics = _topic_set(_tokenize(first_text))
    second_topics = _topic_set(_tokenize(second_text))
    if not first_topics and not second_topics:
        return 0.5
    union = first_topics | second_topics
    intersection = first_topics & second_topics
    base = len(intersection) / len(union) if union else 0.0

    conflict = any(
        (left in first_topics and right in second_topics) or (right in first_topics and left in second_topics)
        for left, right in CONFLICTING_TOPIC_PAIRS
    )
    if conflict:
        base *= 0.65
    return base
def _sequence_similarity(first_text: str, second_text: str) -> float:
    first = preprocess_text(first_text)
    second = preprocess_text(second_text)
    if not first or not second:
        return 0.0
    return SequenceMatcher(None, first, second).ratio()


# Считает текстовую похожесть двух описаний через TF-IDF.
def tfidf_compatibility(first_text: str, second_text: str) -> float:
    prepared = [preprocess_text(first_text), preprocess_text(second_text)]
    if not prepared[0] or not prepared[1]:
        return 0.0

    word_vectorizer = TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, max_df=0.95)
    word_matrix = word_vectorizer.fit_transform(prepared)
    word_similarity = _safe_similarity(word_matrix[0:1], word_matrix[1:2])

    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), sublinear_tf=True)
    char_matrix = char_vectorizer.fit_transform(prepared)
    char_similarity = _safe_similarity(char_matrix[0:1], char_matrix[1:2])

    overlap_similarity = _weighted_jaccard_similarity(first_text, second_text)
    topic_similarity = _topic_similarity(first_text, second_text)
    sequence_similarity = _sequence_similarity(first_text, second_text)

    similarity = (
        word_similarity * 0.38
        + char_similarity * 0.18
        + overlap_similarity * 0.22
        + topic_similarity * 0.17
        + sequence_similarity * 0.05
    )
    return round(float(max(0.0, min(1.0, similarity)) * 100.0), 2)


# Считает TF-IDF оценки для набора кандидатов.
def bulk_tfidf_scores(base_text: str, others: Iterable[str]) -> List[float]:
    return [tfidf_compatibility(base_text, other) for other in others]
