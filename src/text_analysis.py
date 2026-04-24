from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.constants import BASE_RU_STOPWORDS


RU_STOPWORDS = BASE_RU_STOPWORDS
RU_STOPWORDS_SET = set(RU_STOPWORDS)
NON_ALNUM_RE = re.compile(r"[^a-zа-я0-9\s]", flags=re.IGNORECASE)

NORM_MAP = {
    "айти": "it",
    "ит": "it",
    "программист": "it",
    "разработчик": "it",
    "кодер": "it",
    "дев": "it",
    "девопс": "it",
    "qa": "it",
    "тестировщик": "it",
    "аналитик": "it",
    "дизайнер": "design",
    "ux": "design",
    "ui": "design",
    "маркетолог": "marketing",
    "смм": "marketing",
    "smm": "marketing",
    "путешествия": "путешествие",
    "путешествовать": "путешествие",
    "путешествую": "путешествие",
    "поездки": "путешествие",
    "поездка": "путешествие",
    "трип": "путешествие",
    "trip": "путешествие",
    "поход": "путешествие",
    "походы": "путешествие",
    "спортзал": "спорт",
    "зал": "спорт",
    "качалка": "спорт",
    "фитнес": "спорт",
    "бег": "спорт",
    "бегаю": "спорт",
    "йога": "спорт",
    "пилатес": "спорт",
    "плавание": "спорт",
    "плавать": "спорт",
    "велосипед": "спорт",
    "велопрогулки": "спорт",
    "football": "футбол",
    "basketball": "баскетбол",
    "тренировки": "тренировка",
    "тренируюсь": "тренировка",
    "тренироваться": "тренировка",
    "киношка": "кино",
    "кинотеатр": "кино",
    "cinema": "кино",
    "movie": "кино",
    "фильмы": "фильм",
    "сериалы": "сериал",
    "сериалов": "сериал",
    "аниме": "аниме",
    "мультики": "мультфильм",
    "мультфильмы": "мультфильм",
    "документалки": "документальный",
    "книги": "книга",
    "книжки": "книга",
    "чтение": "книга",
    "читать": "книга",
    "литература": "книга",
    "музыка": "музыка",
    "музон": "музыка",
    "слушаю": "музыка",
    "концерты": "концерт",
    "концерт": "концерт",
    "рэп": "hiphop",
    "хипхоп": "hiphop",
    "hip-hop": "hiphop",
    "рок": "rock",
    "рокнролл": "rock",
    "джаз": "jazz",
    "классика": "classical",
    "прогулки": "прогулка",
    "прогулок": "прогулка",
    "гулять": "прогулка",
    "гуляю": "прогулка",
    "парк": "прогулка",
    "набережная": "прогулка",
    "психология": "психолог",
    "психологию": "психолог",
    "психологией": "психолог",
    "общительный": "общение",
    "общительная": "общение",
    "коммуникабельный": "общение",
    "интроверт": "спокойный",
    "экстраверт": "активный",
    "домосед": "домашний",
    "домоседка": "домашний",
    "семья": "семейный",
    "семейный": "семейный",
    "дети": "семейный",
    "ребенок": "семейный",
    "ребенка": "семейный",
    "отношения": "отношения",
    "relationship": "отношения",
    "серьезные": "серьезно",
    "серьезных": "серьезно",
    "брак": "серьезно",
    "жениться": "серьезно",
    "замуж": "серьезно",
    "флирт": "легко",
    "общение": "общение",
    "чат": "общение",
    "переписка": "общение",
    "дружба": "дружба",
    "друзья": "дружба",
    "friendship": "дружба",
    "настолки": "настольные_игры",
    "настольные": "настольные_игры",
    "настольных": "настольные_игры",
    "boardgames": "настольные_игры",
    "игры": "игра",
    "гейминг": "игра",
    "геймер": "игра",
    "gaming": "игра",
    "ps5": "игра",
    "xbox": "игра",
    "комп": "игра",
    "учеба": "учеба",
    "учусь": "учеба",
    "студент": "учеба",
    "универ": "учеба",
    "университет": "учеба",
    "работа": "карьера",
    "работаю": "карьера",
    "карьера": "карьера",
    "бизнес": "карьера",
    "предприниматель": "карьера",
    "кофе": "кофе",
    "кофейни": "кофе",
    "чай": "чай",
    "готовка": "кулинария",
    "готовить": "кулинария",
    "кулинария": "кулинария",
    "выпечка": "кулинария",
    "ресторан": "еда",
    "рестораны": "еда",
    "кафе": "еда",
    "вино": "вино",
    "винцо": "вино",
    "бар": "бар",
    "бары": "бар",
    "тусовки": "вечеринка",
    "вечеринки": "вечеринка",
    "клуб": "вечеринка",
    "клубы": "вечеринка",
    "природа": "природа",
    "лес": "природа",
    "горы": "природа",
    "море": "природа",
    "пляж": "природа",
    "животные": "животные",
    "питомцы": "животные",
    "кот": "кошка",
    "коты": "кошка",
    "кошка": "кошка",
    "кошки": "кошка",
    "собака": "собака",
    "собаки": "собака",
    "пес": "собака",
    "пёс": "собака",
    "волонтер": "волонтерство",
    "волонтерство": "волонтерство",
    "благотворительность": "волонтерство",
    "саморазвитие": "развитие",
    "развитие": "развитие",
    "медитация": "осознанность",
    "mindfulness": "осознанность",
    "юмор": "юмор",
    "шутки": "юмор",
    "мемы": "юмор",
    "мем": "юмор",
    "искренность": "ценности",
    "честность": "ценности",
    "уважение": "ценности",
    "доброта": "ценности",
    "эмпатия": "ценности",
}

RU_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "иями",
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


def _stem_ru(token: str) -> str:
    """Простое отсечение русских суффиксов для нормализации токенов."""
    if token.isdigit() or len(token) < 4:
        return token
    for suffix in RU_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)]
    return token


def _normalize_token(token: str) -> str:
    """Нормализует токен через словарь замен и стемминг."""
    mapped = NORM_MAP.get(token, token)
    return _stem_ru(mapped)


def _tokenize(text: str) -> List[str]:
    """Токенизирует текст и удаляет шумовые/стоп-слова."""
    normalized = text.lower().replace("ё", "е").strip()
    normalized = NON_ALNUM_RE.sub(" ", normalized)
    tokens: List[str] = []
    for tok in normalized.split():
        if not tok:
            continue
        if len(tok) < 2 and not tok.isdigit():
            continue
        tok = _normalize_token(tok)
        if tok in RU_STOPWORDS_SET:
            continue
        tokens.append(tok)
    return tokens


def preprocess_text(text: str) -> str:
    """Возвращает нормализованный текст для векторизации."""
    return " ".join(_tokenize(text))


def _safe_similarity(left_matrix, right_matrix) -> float:
    """Считает cosine similarity и приводит значение к float."""
    value = cosine_similarity(left_matrix, right_matrix)[0][0]
    return float(value)


def _weighted_jaccard_similarity(first_text: str, second_text: str) -> float:
    """Считает взвешенный Жаккар по частотам токенов."""
    first_tokens = _tokenize(first_text)
    second_tokens = _tokenize(second_text)
    if not first_tokens or not second_tokens:
        return 0.0

    left = Counter(first_tokens)
    right = Counter(second_tokens)
    words = set(left) | set(right)
    inter = sum(min(left[w], right[w]) for w in words)
    union = sum(max(left[w], right[w]) for w in words)
    if union == 0:
        return 0.0
    return float(inter / union)


def tfidf_compatibility(first_text: str, second_text: str) -> float:
    """Считает совместимость двух текстов в процентах."""
    prepared = [preprocess_text(first_text), preprocess_text(second_text)]
    if not prepared[0] or not prepared[1]:
        return 0.0

    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.95)
    word_matrix = word_vectorizer.fit_transform(prepared)
    word_similarity = _safe_similarity(word_matrix[0:1], word_matrix[1:2])

    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    char_matrix = char_vectorizer.fit_transform(prepared)
    char_similarity = _safe_similarity(char_matrix[0:1], char_matrix[1:2])

    overlap_similarity = _weighted_jaccard_similarity(first_text, second_text)
    # Формула итогового скоринга: 0.55*word + 0.25*char + 0.2*jaccard.
    similarity = (word_similarity * 0.55) + (char_similarity * 0.25) + (overlap_similarity * 0.2)
    return round(float(similarity * 100.0), 2)


def bulk_tfidf_scores(base_text: str, others: Iterable[str]) -> List[float]:
    """Считает совместимость базового текста с набором других текстов."""
    all_texts = [preprocess_text(base_text)] + [preprocess_text(v) for v in others]
    if not all_texts[0]:
        return [0.0 for _ in all_texts[1:]]

    word_vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_df=0.95)
    word_matrix = word_vectorizer.fit_transform(all_texts)
    word_sims = cosine_similarity(word_matrix[0:1], word_matrix[1:]).flatten()

    char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    char_matrix = char_vectorizer.fit_transform(all_texts)
    char_sims = cosine_similarity(char_matrix[0:1], char_matrix[1:]).flatten()

    overlap_sims = [_weighted_jaccard_similarity(base_text, other) for other in others]

    return [
        round(float(((w * 0.55) + (c * 0.25) + (o * 0.2)) * 100.0), 2)
        for w, c, o in zip(word_sims, char_sims, overlap_sims)
    ]
