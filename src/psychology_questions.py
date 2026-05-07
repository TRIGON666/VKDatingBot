"""
Психологический опросник оценки совместимости.
Охватывает Большую пятёрку, стиль привязанности, стили разрешения конфликтов, языки любви и ценности.
"""

from typing import Dict, List, Optional, Tuple

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

BIG5_DIMENSIONS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")

BIG5_TRAITS = set(BIG5_DIMENSIONS)
ATTACHMENT_TRAITS = {"attachment_secure", "attachment_anxious", "attachment_avoidant"}
CONFLICT_TRAITS = {"conflict_collaborative", "conflict_avoiding", "conflict_competitive"}
LOVE_LANGUAGE_TRAITS = {"love_physical_touch", "love_words", "love_quality_time", "love_acts_service"}
VALUES_TRAITS = {"values_family", "values_independence"}

GROUP_WEIGHTS = {
    "big5": 0.35,
    "attachment": 0.30,
    "conflict": 0.20,
    "love_language": 0.10,
    "values": 0.05,
}

TRAIT_GROUP = {
    **{trait: "big5" for trait in BIG5_TRAITS},
    **{trait: "attachment" for trait in ATTACHMENT_TRAITS},
    **{trait: "conflict" for trait in CONFLICT_TRAITS},
    **{trait: "love_language" for trait in LOVE_LANGUAGE_TRAITS},
    **{trait: "values" for trait in VALUES_TRAITS},
}

# Полный список всех вопросов для последовательного представления.
ALL_QUESTIONS = [question for questions in PSYCHOLOGY_QUESTIONS.values() for question in questions]

# Группы вопросов для бота
QUESTION_GROUPS = {
    "big5": {
        "name": "Личность (Big Five)",
        "description": "Вопросы о вашем темпераменте и характере",
        "count": 10,
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

BIG5_LABELS = {
    "openness": ("Открытость", "интерес к новому, фантазии, идеям и необычному опыту"),
    "conscientiousness": ("Организованность", "планирование, надежность, порядок и ответственность"),
    "extraversion": ("Общительность", "энергия от людей, активность, инициативность в контакте"),
    "agreeableness": ("Доброжелательность", "мягкость, эмпатия, доверие и готовность учитывать другого"),
    "neuroticism": ("Эмоциональная чувствительность", "склонность сильнее переживать стресс и неопределенность"),
}

PERSONALITY_ARCHETYPES = [
    (
        "Спокойный дипломат",
        lambda s: s.get("agreeableness", 0) >= 65 and s.get("neuroticism", 0) < 60,
        "вы мягко общаетесь, цените доверие и обычно стараетесь решать напряжение без давления.",
    ),
    (
        "Организованный партнер",
        lambda s: s.get("conscientiousness", 0) >= 65,
        "вам важны надежность, договоренности и понятные планы; рядом с вами проще строить стабильность.",
    ),
    (
        "Любознательный исследователь",
        lambda s: s.get("openness", 0) >= 65,
        "вам близки новые впечатления, развитие и живой интерес к миру; отношениям нужна свежесть и смысл.",
    ),
    (
        "Социальный инициатор",
        lambda s: s.get("extraversion", 0) >= 65,
        "вам проще проявляться, знакомиться и оживлять общение; важны контакт и взаимная энергия.",
    ),
    (
        "Чуткий наблюдатель",
        lambda s: s.get("neuroticism", 0) >= 65,
        "вы тонко реагируете на атмосферу и сигналы партнера; особенно важны бережность и ясность.",
    ),
]

ATTACHMENT_LABELS = {
    "attachment_secure": ("Безопасный стиль", "легче строить близость, доверять и говорить о потребностях прямо"),
    "attachment_anxious": ("Тревожный стиль", "важны подтверждения внимания, ясность намерений и эмоциональная доступность"),
    "attachment_avoidant": ("Избегающий стиль", "важны личное пространство, постепенность и уважение к самостоятельности"),
}

CONFLICT_LABELS = {
    "conflict_collaborative": ("Сотрудничество", "вы чаще ищете решение, которое учитывает обе стороны"),
    "conflict_avoiding": ("Избегание", "вам может быть проще отложить спор, чем сразу входить в напряжение"),
    "conflict_competitive": ("Прямое отстаивание", "вы склонны защищать позицию и быстро обозначать несогласие"),
}

LOVE_LANGUAGE_LABELS = {
    "love_physical_touch": ("прикосновения", "объятия, близость и физическая нежность"),
    "love_words": ("слова поддержки", "похвала, теплые сообщения и прямые признания"),
    "love_quality_time": ("время вместе", "внимание, совместные планы и включенность"),
    "love_acts_service": ("забота делами", "помощь, поступки и практическая поддержка"),
}

VALUE_LABELS = {
    "values_family": ("семья и близость", "важны надежные связи, дом и круг близких людей"),
    "values_independence": ("самостоятельность и развитие", "важны свобода выбора, рост и личные цели"),
}


# Работает с вопросами анкеты или теста.
def get_total_questions() -> int:
    """Получить общее количество вопросов в анкете."""
    return len(ALL_QUESTIONS)


# Работает с вопросами анкеты или теста.
def get_questions_by_group(group_name: str) -> List[Dict]:
    """Получить вопросы для определённого психологического измерения."""
    if group_name == "big5":
        return [question for dimension in BIG5_DIMENSIONS for question in PSYCHOLOGY_QUESTIONS[dimension]]
    return PSYCHOLOGY_QUESTIONS.get(group_name, [])


# Работает с вопросами анкеты или теста.
def get_next_question(current_index: int) -> Tuple[Optional[int], Dict]:
    """
    Получить следующий вопрос по порядку.
    
    Параметры:
        current_index: текущий индекс вопроса, начиная с 0.
    
    Возвращает:
        (next_index, question_dict) или (None, {}), если вопросы закончились.
    """
    if current_index >= len(ALL_QUESTIONS) - 1:
        return None, {}
    next_index = current_index + 1
    return next_index, ALL_QUESTIONS[next_index]


# Работает с вопросами анкеты или теста.
def get_question_by_id(question_id: str) -> Dict:
    """Получить вопрос по его идентификатору."""
    for question in ALL_QUESTIONS:
        if question["id"] == question_id:
            return question
    return {}


# Считает психологические шкалы по ответам пользователя.
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


# Формирует понятную расшифровку Big Five для анкеты.
def describe_personality(scores: Dict[str, float]) -> str:
    personality_scores = {key: scores.get(key, 0) for key in BIG5_DIMENSIONS if key in scores}
    if not personality_scores:
        return "Личность: данных пока недостаточно."

    matches = [
        (name, description)
        for name, predicate, description in PERSONALITY_ARCHETYPES
        if predicate(personality_scores)
    ]
    if len(matches) == 1:
        archetype_name, archetype_text = matches[0]
    elif len(matches) > 1:
        archetype_name = "Смешанный тип"
        archetype_text = "выражено несколько сильных черт; ниже — ведущие из них."
    else:
        archetype_name = "Сбалансированный тип"
        archetype_text = "ваши черты распределены ровно; вы можете гибко подстраиваться под ситуацию и партнера."

    strongest = sorted(personality_scores.items(), key=lambda item: item[1], reverse=True)[:2]
    trait_lines = []
    for key, value in strongest:
        label, meaning = BIG5_LABELS[key]
        trait_lines.append(f"- {label}: {value:.0f}% — {meaning}.")

    return "\n".join(
        [
            f"🎭 Личность: {archetype_name}",
            f"Что это значит: {archetype_text}",
            "Сильнее всего выражено:",
            *trait_lines,
        ]
    )


# Выбирает самую выраженную шкалу и ее описание.
def _top_descriptions(scores: Dict[str, float], labels: Dict[str, Tuple[str, str]]) -> List[Tuple[str, str, float]]:
    available = [(key, scores.get(key, 0)) for key in labels if key in scores]
    if not available:
        return []
    max_value = max(value for _, value in available)
    top = [key for key, value in available if value == max_value]
    # Среднее от всех стилей в группе для показа общей "силы выраженности"
    avg_all = sum(scores[key] for key in labels if key in scores) / len(labels) if labels else max_value
    return [(labels[key][0], labels[key][1], float(avg_all)) for key in top]


# Считает психологическую совместимость двух профилей.
def get_compatibility_recommendation(
    user1_scores: Dict[str, float],
    user2_scores: Dict[str, float],
) -> Dict:
    """
    Оценить психологическую совместимость между двумя пользователями.
    
    Параметры:
        user1_scores: Оценки черт из calculate_scores()
        user2_scores: Оценки черт из calculate_scores()
    
    Возвращает:
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
    group_diffs: Dict[str, List[float]] = {}
    
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

            group = TRAIT_GROUP.get(trait)
            if group:
                group_diffs.setdefault(group, []).append(diff)
    
    compared_traits = len(trait_analysis)
    if compared_traits == 0:
        return {
            "overall_score": None,
            "trait_analysis": {},
            "total_traits_compared": 0,
            "message": "Нет пересекающихся психологических метрик",
        }

    # Общая оценка с учетом весов групп.
    group_avg = {
        group: (sum(diffs) / len(diffs))
        for group, diffs in group_diffs.items()
        if diffs
    }
    weight_sum = sum(GROUP_WEIGHTS.get(group, 0.0) for group in group_avg)
    if weight_sum <= 0:
        avg_diff = total_diff / compared_traits
    else:
        avg_diff = sum(group_avg[group] * GROUP_WEIGHTS.get(group, 0.0) for group in group_avg) / weight_sum
    overall_score = max(0, 100 - avg_diff)  # 0-100
    
    return {
        "overall_score": round(overall_score),
        "trait_analysis": trait_analysis,
        "group_avg_diff": {group: round(value, 2) for group, value in group_avg.items()},
        "group_weights": {group: GROUP_WEIGHTS.get(group, 0.0) for group in group_avg},
        "total_traits_compared": compared_traits,
    }


# Формирует краткое описание психологического профиля.
def format_profile_summary(scores: Dict[str, float]) -> str:
    """
    Отобразить психологические результаты в читаемом резюме.
    
    Параметры:
        scores: Оценки из calculate_scores()
    
    Возвращает:
        Отформатированная строка для вывода
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
        lines.append(describe_personality(scores))
    
    if attachment_traits:
        attachment = _top_descriptions(scores, ATTACHMENT_LABELS)
        if attachment:
            value = attachment[0][2]
            if len(attachment) == 1:
                name, description, _ = attachment[0]
                lines.append(f"💕 Стиль отношений: {name} ({value:.0f}%)")
                lines.append(f"   Это значит: {description}.")
            else:
                lines.append(f"💕 Стиль отношений: смешанный ({value:.0f}%)")
                for name, description, _ in attachment:
                    lines.append(f"   {name}: {description}.")
    
    if conflict_traits:
        conflict = _top_descriptions(scores, CONFLICT_LABELS)
        if conflict:
            value = conflict[0][2]
            if len(conflict) == 1:
                name, description, _ = conflict[0]
                lines.append(f"⚖️ В конфликтах: {name} ({value:.0f}%)")
                lines.append(f"   Это значит: {description}.")
            else:
                lines.append(f"⚖️ В конфликтах: смешанный ({value:.0f}%)")
                for name, description, _ in conflict:
                    lines.append(f"   {name}: {description}.")

    love_language = _top_descriptions(scores, LOVE_LANGUAGE_LABELS)
    if love_language:
        value = love_language[0][2]
        if len(love_language) == 1:
            name, description, _ = love_language[0]
            lines.append(f"💌 Язык любви: {name} ({value:.0f}%)")
            lines.append(f"   Особенно считываются: {description}.")
        else:
            lines.append(f"💌 Язык любви: смешанный ({value:.0f}%)")
            for name, description, _ in love_language:
                lines.append(f"   {name}: {description}.")

    value_focus = _top_descriptions(scores, VALUE_LABELS)
    if value_focus:
        value = value_focus[0][2]
        if len(value_focus) == 1:
            name, description, _ = value_focus[0]
            lines.append(f"🧭 Ценности: {name} ({value:.0f}%)")
            lines.append(f"   В отношениях: {description}.")
        else:
            lines.append(f"🧭 Ценности: смешанные ({value:.0f}%)")
            for name, description, _ in value_focus:
                lines.append(f"   {name}: {description}.")
    
    return "\n".join(lines)
