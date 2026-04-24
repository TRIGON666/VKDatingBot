"""
Психологический опросник оценки совместимости.
Охватывает Большую пятёрку, стиль привязанности, стили разрешения конфликтов, языки любви и ценности.
"""

from typing import Dict, List, Tuple

# Наборы вопросов, организованные по психологическим измерениям
PSYCHOLOGY_QUESTIONS = {
    # Большая пятёрка личности (вес 35%) - 8 вопросов
    "openness": [
        {
            "id": "big5_open_1",
            "text": "Я люблю пробовать новые виды деятельности и получать новый опыт",
            "scale": "agree",  # Шкала 1-5: полностью не согласен до полностью согласен
            "trait": "openness",
        },
        {
            "id": "big5_open_2",
            "text": "Я предпочитаю рутину и привычные вещи",
            "scale": "agree",
            "trait": "openness",
            "reverse": True,  # Обратная кодировка
        },
    ],
    "conscientiousness": [
        {
            "id": "big5_cons_1",
            "text": "Я планирую и организую свои дела заранее",
            "scale": "agree",
            "trait": "conscientiousness",
        },
        {
            "id": "big5_cons_2",
            "text": "Я часто перепроверяю свою работу перед тем, как её отправить",
            "scale": "agree",
            "trait": "conscientiousness",
        },
    ],
    "extraversion": [
        {
            "id": "big5_extr_1",
            "text": "В больших компаниях я чувствую себя напряженно",
            "scale": "agree",
            "trait": "extraversion",
            "reverse": True,
        },
        {
            "id": "big5_extr_2",
            "text": "Я люблю быть в центре внимания и общаться с людьми",
            "scale": "agree",
            "trait": "extraversion",
        },
    ],
    "agreeableness": [
        {
            "id": "big5_agr_1",
            "text": "Я считаю, что люди в целом хорошие и достойны доверия",
            "scale": "agree",
            "trait": "agreeableness",
        },
        {
            "id": "big5_agr_2",
            "text": "Я сочувствую людям в трудных ситуациях",
            "scale": "agree",
            "trait": "agreeableness",
        },
    ],
    "neuroticism": [
        {
            "id": "big5_neur_1",
            "text": "Я часто беспокоюсь о том, что может пойти не так",
            "scale": "agree",
            "trait": "neuroticism",
        },
        {
            "id": "big5_neur_2",
            "text": "Я быстро раздражаюсь, когда дела не идут по плану",
            "scale": "agree",
            "trait": "neuroticism",
        },
    ],
    # Стиль привязанности взрослого (вес 30%) - 6 вопросов
    "attachment": [
        {
            "id": "attach_1",
            "text": "Я комфортно чувствую себя в близких отношениях и люблю быть рядом с партнёром",
            "scale": "agree",
            "trait": "attachment_secure",
        },
        {
            "id": "attach_2",
            "text": "Я часто беспокоюсь, что мой партнёр не ценит меня так же, как я его",
            "scale": "agree",
            "trait": "attachment_anxious",
        },
        {
            "id": "attach_3",
            "text": "Я предпочитаю независимость и дистанцию в отношениях",
            "scale": "agree",
            "trait": "attachment_avoidant",
        },
        {
            "id": "attach_4",
            "text": "Я верю, что отношения могут быть стабильными и надёжными",
            "scale": "agree",
            "trait": "attachment_secure",
        },
        {
            "id": "attach_5",
            "text": "Мне сложно открыться и показать уязвимость",
            "scale": "agree",
            "trait": "attachment_avoidant",
        },
        {
            "id": "attach_6",
            "text": "Я нуждаюсь в частых подтверждениях внимания и любви",
            "scale": "agree",
            "trait": "attachment_anxious",
        },
    ],
    # Стили разрешения конфликтов и общения (вес 20%) - 5 вопросов
    "conflict": [
        {
            "id": "conf_1",
            "text": "Когда мы не согласны, я предпочитаю обсудить проблему спокойно и найти компромисс",
            "scale": "agree",
            "trait": "conflict_collaborative",
        },
        {
            "id": "conf_2",
            "text": "Я избегаю конфликтов, даже если это означает отойти от важных вопросов",
            "scale": "agree",
            "trait": "conflict_avoiding",
        },
        {
            "id": "conf_3",
            "text": "Я настаиваю на своей позиции и пытаюсь победить в споре",
            "scale": "agree",
            "trait": "conflict_competitive",
        },
        {
            "id": "conf_4",
            "text": "Я готов попробовать решение, которое удовлетворит обе стороны",
            "scale": "agree",
            "trait": "conflict_collaborative",
        },
        {
            "id": "conf_5",
            "text": "Я прислушиваюсь к мнению партнёра, даже если не согласен",
            "scale": "agree",
            "trait": "conflict_collaborative",
        },
    ],
    # Языки любви (вес 10%) - 4 вопроса
    "love_language": [
        {
            "id": "love_1",
            "text": "Мне нравятся физические проявления привязанности (объятия, прикосновения)",
            "scale": "agree",
            "trait": "love_physical_touch",
        },
        {
            "id": "love_2",
            "text": "Я ценю похвалу и слова поддержки",
            "scale": "agree",
            "trait": "love_words",
        },
        {
            "id": "love_3",
            "text": "Я хочу, чтобы партнёр проводил со мной время и уделял мне внимание",
            "scale": "agree",
            "trait": "love_quality_time",
        },
        {
            "id": "love_4",
            "text": "Мне нравятся практические проявления заботы (помощь, подарки, дела)",
            "scale": "agree",
            "trait": "love_acts_service",
        },
    ],
    # Ценности и жизненные цели (вес 5%) - 2 вопроса
    "values": [
        {
            "id": "val_1",
            "text": "Для меня очень важна семья и близкие люди",
            "scale": "agree",
            "trait": "values_family",
        },
        {
            "id": "val_2",
            "text": "Я ценю самостоятельность и личное развитие выше всего",
            "scale": "agree",
            "trait": "values_independence",
        },
    ],
}

# Полный список всех вопросов для последовательного представления
ALL_QUESTIONS = []
for dimension, questions in PSYCHOLOGY_QUESTIONS.items():
    ALL_QUESTIONS.extend(questions)

# Группы вопросов для бота
QUESTION_GROUPS = {
    "big5": {
        "name": "Личность (Big Five)",
        "description": "Вопросы о вашем темпераменте и характере",
        "count": 8,
    },
    "attachment": {
        "name": "Стиль отношений",
        "description": "Как вы относитесь к близким отношениям",
        "count": 6,
    },
    "conflict": {
        "name": "Разрешение конфликтов",
        "description": "Как вы справляетесь с несогласием",
        "count": 5,
    },
    "love_language": {
        "name": "Языки любви",
        "description": "Как вы выражаете и получаете любовь",
        "count": 4,
    },
    "values": {
        "name": "Ценности и цели",
        "description": "Что важно для вас в жизни",
        "count": 2,
    },
}


def get_total_questions() -> int:
    """Получить общее количество вопросов в анкете."""
    return len(ALL_QUESTIONS)


def get_questions_by_group(group_name: str) -> List[Dict]:
    """Получить вопросы для определённого психологического измерения."""
    if group_name == "big5":
        return PSYCHOLOGY_QUESTIONS["openness"] + PSYCHOLOGY_QUESTIONS["conscientiousness"] + \
               PSYCHOLOGY_QUESTIONS["extraversion"] + PSYCHOLOGY_QUESTIONS["agreeableness"] + \
               PSYCHOLOGY_QUESTIONS["neuroticism"]
    return PSYCHOLOGY_QUESTIONS.get(group_name, [])


def get_next_question(current_index: int) -> Tuple[int, Dict]:
    """
    Get the next question in sequence.
    
    Args:
        current_index: Current question index (0-based)
    
    Returns:
        Tuple of (next_index, question_dict) or (None, {}) if finished
    """
    if current_index >= len(ALL_QUESTIONS) - 1:
        return None, {}
    next_index = current_index + 1
    return next_index, ALL_QUESTIONS[next_index]


def get_question_by_id(question_id: str) -> Dict:
    """Получить вопрос по его идентификатору."""
    for question in ALL_QUESTIONS:
        if question["id"] == question_id:
            return question
    return {}


def calculate_scores(answers: Dict[str, int]) -> Dict[str, float]:
    """
    Calculate trait scores from raw answers (1-5 scale).
    
    Args:
        answers: Dict mapping question_id to answer (1-5)
    
    Returns:
        Dict with trait names and scores (0-100)
    """
    trait_scores = {}
    trait_counts = {}
    
    for q_id, answer in answers.items():
        question = get_question_by_id(q_id)
        if not question:
            continue
        
        trait = question.get("trait")
        if not trait:
            continue
        
        # Обработка обратно кодированных пунктов
        score = answer
        if question.get("reverse"):
            score = 6 - answer  # Обратная кодировка: 5->1, 4->2, 3->3, 2->4, 1->5
        
        if trait not in trait_scores:
            trait_scores[trait] = 0
            trait_counts[trait] = 0
        
        trait_scores[trait] += score
        trait_counts[trait] += 1
    
    # Нормализация на шкалу 0-100
    normalized = {}
    for trait, total in trait_scores.items():
        count = trait_counts[trait]
        avg = total / count if count > 0 else 0
        # Преобразование шкалы 1-5 в 0-100
        normalized[trait] = round((avg - 1) * 25)  # 1->0, 3->50, 5->100
    
    return normalized


def get_compatibility_recommendation(
    user1_scores: Dict[str, float],
    user2_scores: Dict[str, float],
) -> Dict:
    """
    Охслюдить психологическую совместимость между двумя пользователями.
    
    Параметры:
        user1_scores: Оценки черт из calculate_scores()
        user2_scores: Оценки черт из calculate_scores()
    
    Восвращено:
        Словарь с метриками и рекомендациями совместимости
    """
    if not user1_scores or not user2_scores:
        return {
            "overall_score": None,
            "trait_analysis": {},
            "total_traits_compared": 0,
            "message": "Недостаточно данных",
        }
    
    total_diff = 0
    trait_analysis = {}
    
    # Сравнить каждую черту
    for trait in user1_scores:
        if trait in user2_scores:
            diff = abs(user1_scores[trait] - user2_scores[trait])
            total_diff += diff
            
            # Классифицирование совместимости для этой черты
            if diff < 20:
                compatibility = "высокая совместимость"
            elif diff < 40:
                compatibility = "умеренная совместимость"
            else:
                compatibility = "потребуется компромисс"
            
            trait_analysis[trait] = {
                "difference": diff,
                "compatibility": compatibility,
            }
    
    compared_traits = len(trait_analysis)
    if compared_traits == 0:
        return {
            "overall_score": None,
            "trait_analysis": {},
            "total_traits_compared": 0,
            "message": "Нет пересекающихся психологических метрик",
        }

    # Общая оценка (чем меньше разница = выше совместимость)
    avg_diff = total_diff / compared_traits
    overall_score = max(0, 100 - avg_diff)  # 0-100
    
    return {
        "overall_score": round(overall_score),
        "trait_analysis": trait_analysis,
        "total_traits_compared": compared_traits,
    }


def format_profile_summary(scores: Dict[str, float]) -> str:
    """
    Отображают психологические результаты в читаемом резюме.
    
    Параметры:
        scores: Оценки из calculate_scores()
    
    Восвращено:
        Отоформатная строка для вывода
    """
    if not scores:
        return "Опросник не заполнен"
    
    lines = ["📊 Психологический профиль:\n"]
    
    # Группировка по измерениям
    personality_keys = {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"}
    personality_traits = {k: v for k, v in scores.items() if k in personality_keys}
    attachment_traits = {k: v for k, v in scores.items() if k.startswith("attachment")}
    conflict_traits = {k: v for k, v in scores.items() if k.startswith("conflict")}
    
    if personality_traits:
        avg = sum(personality_traits.values()) / len(personality_traits)
        lines.append(f"🎭 Личность: {avg:.0f}%")
    
    if attachment_traits:
        secure = scores.get("attachment_secure", 0)
        anxious = scores.get("attachment_anxious", 0)
        avoidant = scores.get("attachment_avoidant", 0)
        if secure > anxious and secure > avoidant:
            style = "Безопасный стиль"
        elif anxious > avoidant:
            style = "Тревожный стиль"
        else:
            style = "Избегающий стиль"
        lines.append(f"💕 Стиль отношений: {style}")
    
    if conflict_traits:
        collab = scores.get("conflict_collaborative", 0)
        avoid = scores.get("conflict_avoiding", 0)
        compet = scores.get("conflict_competitive", 0)
        if collab > avoid and collab > compet:
            style = "Сотрудничество"
        elif avoid > compet:
            style = "Избегание"
        else:
            style = "Конкуренция"
        lines.append(f"⚖️ Разрешение конфликтов: {style}")
    
    return "\n".join(lines)
