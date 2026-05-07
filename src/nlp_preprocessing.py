"""Улучшенная предварительная обработка русского текста для NLP."""

from __future__ import annotations

import re
import unicodedata
import logging
from collections import Counter
from typing import List

from src.constants import BASE_RU_STOPWORDS


logger = logging.getLogger(__name__)

try:
    import pymorphy2
    morph = pymorphy2.MorphAnalyzer()
    HAS_PYMORPHY = True
except ImportError:
    HAS_PYMORPHY = False


RU_STOPWORDS = BASE_RU_STOPWORDS | {
    "очень",
    "более",
    "еще",
    "то",
    "вот",
    "просто",
    "чем",
    "так",
    "уж",
    "вы",
    "никто",
    "никогда",
    "ничего",
    "там",
    "тогда",
    "тут",
    "здесь",
}


NON_WORD_KEEP_PUNCT_RE = re.compile(r"[^\w\s\-\.\,\!\?\:]+", flags=re.UNICODE)
MULTI_SPACE_RE = re.compile(r"\s+")
REPEATED_CYRILLIC_RE = re.compile(r"([а-яё])\1{2,}")
NON_WORD_RE = re.compile(r"[^\w]", flags=re.UNICODE)


# Нормализует текст профиля перед NLP-анализом.
def preprocess_profile(text: str) -> str:
    """
    Полная предварительная обработка текста профиля.
    
    1. Привести к нижнему регистру
    2. Удалить эмодзи и спецсимволы
    3. Нормализовать пробелы
    4. Применить лемматизацию (если доступна pymorphy2)
    5. Удалить стоп-слова
    """
    if not text:
        return ""
    
    # 1. Привести к нижнему регистру
    text = text.lower()
    
    # 2. Нормализовать Unicode и удалить эмодзи
    text = unicodedata.normalize("NFKD", text)
    text = NON_WORD_KEEP_PUNCT_RE.sub('', text)
    
    # 3. Нормализовать пробелы
    text = MULTI_SPACE_RE.sub(' ', text).strip()
    text = REPEATED_CYRILLIC_RE.sub(r'\1', text)  # Убрать повторы букв
    
    # 4. Применить лемматизацию если доступна pymorphy2
    if HAS_PYMORPHY:
        words = text.split()
        lemmas = []
        for word in words:
            word_clean = NON_WORD_RE.sub('', word)
            if word_clean and len(word_clean) > 2:
                try:
                    lemma = morph.parse(word_clean)[0].normal_form
                    lemmas.append(lemma)
                except Exception:
                    logger.debug("Failed to lemmatize token, using raw token fallback", exc_info=True)
                    lemmas.append(word_clean)
        text = ' '.join(lemmas)
    
    # 5. Удалить стоп-слова и короткие слова
    words = text.split()
    filtered = [w for w in words if w not in RU_STOPWORDS and len(w) > 2]
    
    return ' '.join(filtered)


# Выделяет ключевые слова из подготовленного текста.
def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """Извлечь ключевые слова из профиля."""
    text = preprocess_profile(text)
    words = text.split()
    words = [w for w in words if len(w) > 3]
    
    if not words:
        return []
    
    counter = Counter(words)
    return [word for word, _ in counter.most_common(top_n)]
