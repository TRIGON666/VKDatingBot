from __future__ import annotations

import csv
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import sys

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CITIES = [
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Красноярск",
    "Челябинск",
    "Самара",
    "Уфа",
    "Ростов-на-Дону",
    "Краснодар",
    "Омск",
    "Воронеж",
    "Пермь",
    "Волгоград",
    "Саратов",
    "Тюмень",
    "Тольятти",
    "Барнаул",
    "Ижевск",
    "Махачкала",
    "Хабаровск",
    "Владивосток",
    "Ярославль",
    "Оренбург",
    "Томск",
    "Кемерово",
    "Новокузнецк",
    "Рязань",
    "Астрахань",
    "Пенза",
    "Калининград",
    "Тула",
]

GENDERS = ["Мужской", "Женский"]
AGE_RANGE = (18, 42)

PHRASES: Dict[str, List[str]] = {
    "active_extreme": [
        "Не представляю жизнь без движения и новых впечатлений.",
        "По выходным меня чаще всего можно найти в горах или на трассе.",
        "Люблю адреналин, длинные маршруты и спонтанные выезды.",
        "Спорт для меня - это не раз в неделю, а образ жизни.",
        "Вечно ищу приключения и не люблю сидеть на месте.",
        "Если есть возможность сорваться в поездку, я почти всегда за.",
        "Мне ближе активный ритм, чем спокойные домашние вечера.",
        "Предпочитаю живые эмоции и насыщенный темп.",
    ],
    "gym_fitness": [
        "Спортзал у меня по расписанию, без него день не засчитывается.",
        "Слежу за питанием, стараюсь держать режим и форму.",
        "Люблю железо, кардио и понятный прогресс.",
        "Тренировки для меня - отличный способ перезагрузки.",
        "Регулярность в спорте для меня важнее, чем редкие подвиги.",
        "Ищу человека, который тоже уважает здоровье и дисциплину.",
        "Нормально отношусь к режиму, БЖУ и ранним подъемам.",
        "Больше всего ценю энергию, выносливость и хорошее самочувствие.",
    ],
    "homebody_geek": [
        "Я скорее домашний человек: книги, игры, фильмы и чай.",
        "Лучший вечер для меня - это тишина, диван и любимый сериал.",
        "Люблю спокойные хобби, настолки и уютные разговоры.",
        "Не фанат шумных компаний, мне комфортнее в маленьком кругу.",
        "В свободное время могу залипнуть в игру или научпоп.",
        "Интровертный вайб мне ближе, чем бесконечные тусовки.",
        "Люблю погружаться в хобби глубоко и без лишнего шума.",
        "Идеальный отдых для меня - домашний и без суеты.",
    ],
    "party_social": [
        "Я люблю живые встречи, общение и новые знакомства.",
        "По пятницам чаще всего тянет в бар или на вечеринку.",
        "Мне комфортно там, где много людей, музыки и движения.",
        "Обожаю спонтанные планы и компанию друзей.",
        "Тишина быстро надоедает, а вот атмосфера города - нет.",
        "Нормально чувствую себя в шумных компаниях и на танцполе.",
        "Часто выбираю события, где можно познакомиться с новыми людьми.",
        "Люблю, когда вокруг жизнь, а не просто фон.",
    ],
    "family_serious": [
        "Серьезные отношения для меня важнее мимолетных историй.",
        "Ценю честность, надежность и спокойное будущее.",
        "Хочу построить отношения, в которых есть опора и уважение.",
        "Для меня важны семья, доверие и понятные планы.",
        "Не люблю пустые обещания, ищу взрослый подход.",
        "Смотрю на отношения всерьез и без лишней игры.",
        "Комфортно чувствую себя в стабильности и ясности.",
        "Хочу человека, с которым можно думать о будущем.",
    ],
    "casual_fwb": [
        "Пока не готов к серьезным обязательствам и сложным ожиданиям.",
        "Предпочитаю легкий формат без лишнего давления.",
        "Сейчас больше ценю свободу и простоту в общении.",
        "Не хочу торопить события и строить громкие планы.",
        "Мне ближе честный, но ненапряженный формат.",
        "Хочу общение без лишнего драматизма и претензий.",
        "Смотрю на знакомства спокойно и без спешки.",
        "Нормально отношусь к свободному формату, если все честно.",
    ],
    "career_hustle": [
        "Сейчас у меня много работы, проектов и созвонов.",
        "Мне важно расти в карьере и не терять темп.",
        "Люблю амбициозных людей и сильный рабочий драйв.",
        "Часто думаю про цели, развитие и новые возможности.",
        "Работа занимает много времени, но мне это подходит.",
        "Ценю людей, которые умеют держать фокус и не ленятся.",
        "В ближайшее время хочу сделать рывок в профессии.",
        "Мне интересны деньги, рост и понятный результат.",
    ],
    "creative_art": [
        "Мне близки искусство, музыка и визуальная эстетика.",
        "Часто хожу на выставки, концерты и люблю кино.",
        "Нравится замечать детали и смотреть на мир иначе.",
        "Творческая атмосфера меня сильно заряжает.",
        "Могу часами обсуждать стиль, музыку и визуал.",
        "Вдохновение для меня важно почти так же, как режим.",
        "Люблю людей с воображением и вкусом к жизни.",
        "Меня легко увлечь идеей, образом или настроением.",
    ],
    "travel_nomad": [
        "Путешествия для меня - не редкость, а привычка.",
        "Мне легко собраться и уехать в новый город или страну.",
        "Удаленка и свобода перемещения меня очень привлекают.",
        "Люблю менять картинку и редко засиживаюсь на одном месте.",
        "Мне нравится жить динамично и без жесткой привязки к локации.",
        "Дорога и новые места для меня почти как ресурс.",
        "Ищу человека, который нормально относится к спонтанным поездкам.",
        "Постоянное движение для меня естественно.",
    ],
    "pet_lover": [
        "У меня дома есть питомец, и он давно главный в семье.",
        "Животные мне очень близки, без них как-то пусто.",
        "Люблю долгие прогулки с собакой и спокойную заботу.",
        "Мне нравятся люди, которые тепло относятся к животным.",
        "Почти в каждом разговоре рано или поздно всплывают питомцы.",
        "Считаю, что животные делают дом живым.",
        "Если у тебя тоже есть любимец, это большой плюс.",
        "Хочется отношений с добротой и без жесткости.",
    ],
    "generic": [
        "Привет, я тут без лишнего пафоса.",
        "Обычный человек, работаю и стараюсь жить спокойно.",
        "Люблю вкусную еду, нормальный сон и адекватных людей.",
        "Пока просто смотрю, что тут и как.",
        "Слушаю музыку, гуляю и не люблю лишнюю суету.",
        "В целом я за простое и понятное общение.",
        "Не хочу писать много, лучше расскажу в личке.",
        "Мне нравится, когда все по-человечески и без игры.",
    ],
}

CATEGORY_META: Dict[str, Dict[str, object]] = {
    "active_extreme": {"tempo": "быстрый", "social": "высокая", "openness": "высокая"},
    "gym_fitness": {"tempo": "средний", "social": "средняя", "openness": "средняя"},
    "homebody_geek": {"tempo": "медленный", "social": "низкая", "openness": "средняя"},
    "party_social": {"tempo": "быстрый", "social": "высокая", "openness": "высокая"},
    "family_serious": {"tempo": "средний", "social": "средняя", "openness": "средняя"},
    "casual_fwb": {"tempo": "средний", "social": "средняя", "openness": "низкая"},
    "career_hustle": {"tempo": "быстрый", "social": "средняя", "openness": "средняя"},
    "creative_art": {"tempo": "средний", "social": "средняя", "openness": "высокая"},
    "travel_nomad": {"tempo": "быстрый", "social": "высокая", "openness": "высокая"},
    "pet_lover": {"tempo": "средний", "social": "средняя", "openness": "средняя"},
    "generic": {"tempo": "средний", "social": "средняя", "openness": "средняя"},
}

MATCH_RULES: List[Tuple[str, str, str]] = [
    ("active_extreme", "active_extreme", "positive"),
    ("active_extreme", "travel_nomad", "positive"),
    ("active_extreme", "gym_fitness", "positive"),
    ("gym_fitness", "gym_fitness", "positive"),
    ("homebody_geek", "homebody_geek", "positive"),
    ("homebody_geek", "family_serious", "positive"),
    ("homebody_geek", "creative_art", "positive"),
    ("party_social", "party_social", "positive"),
    ("party_social", "casual_fwb", "positive"),
    ("family_serious", "family_serious", "positive"),
    ("family_serious", "pet_lover", "positive"),
    ("casual_fwb", "casual_fwb", "positive"),
    ("career_hustle", "career_hustle", "positive"),
    ("career_hustle", "gym_fitness", "positive"),
    ("creative_art", "creative_art", "positive"),
    ("travel_nomad", "travel_nomad", "positive"),
    ("pet_lover", "pet_lover", "positive"),
    ("family_serious", "casual_fwb", "negative"),
    ("homebody_geek", "party_social", "negative"),
    ("active_extreme", "homebody_geek", "negative"),
    ("career_hustle", "casual_fwb", "negative"),
    ("career_hustle", "generic", "negative"),
    ("travel_nomad", "family_serious", "negative"),
    ("party_social", "family_serious", "negative"),
    ("active_extreme", "generic", "negative"),
    ("creative_art", "casual_fwb", "negative"),
    ("pet_lover", "party_social", "negative"),
]

CITY_ALIASES = {
    "москва": ["Москва", "Мск", "Moscow"],
    "санкт петербург": ["Санкт-Петербург", "СПб", "Питер"],
    "новосибирск": ["Новосибирск", "Нск"],
    "екатеринбург": ["Екатеринбург", "Екб"],
    "казань": ["Казань", "Kazan"],
    "нижний новгород": ["Нижний Новгород", "НН"],
    "красноярск": ["Красноярск"],
    "челябинск": ["Челябинск"],
    "самара": ["Самара"],
    "уфа": ["Уфа"],
    "ростов на дону": ["Ростов-на-Дону", "Ростов"],
    "краснодар": ["Краснодар"],
    "омск": ["Омск"],
    "воронеж": ["Воронеж"],
    "пермь": ["Пермь"],
    "волгоград": ["Волгоград"],
    "саратов": ["Саратов"],
    "тюмень": ["Тюмень"],
    "тольятти": ["Тольятти"],
    "барнаул": ["Барнаул"],
    "ижевск": ["Ижевск"],
    "махачкала": ["Махачкала"],
    "хабаровск": ["Хабаровск"],
    "владивосток": ["Владивосток"],
    "ярославль": ["Ярославль"],
    "оренбург": ["Оренбург"],
    "томск": ["Томск"],
    "кемерово": ["Кемерово"],
    "новокузнецк": ["Новокузнецк"],
    "рязань": ["Рязань"],
    "астрахань": ["Астрахань"],
    "пенза": ["Пенза"],
    "калининград": ["Калининград"],
    "тула": ["Тула"],
}

NICK_PARTS = [
    "solar",
    "north",
    "quiet",
    "river",
    "urban",
    "forest",
    "ember",
    "violet",
    "pixel",
    "zen",
    "orbit",
    "nova",
    "wild",
    "meter",
    "cloud",
]

GOALS = [
    "ищу серьезные отношения",
    "важны уважение и поддержка",
    "хочу найти человека для совместной жизни",
    "интересно спокойное общение и встречи",
    "ищу партнера с похожими взглядами",
    "хочется общения без лишнего давления",
    "смотрю на знакомства спокойно и без спешки",
]

HOBBIES = [
    "читаю книги",
    "люблю прогулки",
    "занимаюсь спортом",
    "смотрю фильмы",
    "слушаю музыку",
    "путешествую по выходным",
    "играю в настолки",
    "готовлю дома",
    "хожу в походы",
    "интересуюсь психологией",
    "посещаю выставки",
    "катаюсь на велосипеде",
    "изучаю языки",
    "работаю в it",
    "люблю животных",
    "играю в волейбол",
    "хожу на концерты",
    "увлекаюсь фотографией",
    "слежу за технологиями",
    "люблю кофейни",
    "учу французский",
    "занимаюсь бегом",
    "смотрю документалки",
    "интересуюсь историей",
    "слушаю подкасты",
    "могу часами обсуждать кино",
    "пишу заметки и идеи",
]

PERSONALITY_TEMPLATES = [
    "По характеру я {communication}, а мой ритм жизни {lifestyle}.",
    "Если кратко, я {communication} и живу в {lifestyle} темпе.",
    "Обычно веду себя {communication}, предпочитаю {lifestyle} формат жизни.",
    "Мне ближе {communication} общение и {lifestyle} образ жизни.",
]

VALUES_TEMPLATES = [
    "Для меня важнее всего {values}.",
    "Стараюсь строить жизнь вокруг ценности {values}.",
    "В приоритете у меня {values}.",
    "Мои ориентиры - {values}.",
]

HOBBY_TEMPLATES = [
    "В свободное время {hobbies}.",
    "Чаще всего я {hobbies}.",
    "Обычно после работы {hobbies}.",
    "Мой досуг выглядит так: {hobbies}.",
]

STYLE_TEMPLATES = [
    "По вайбу у меня {style} стиль общения.",
    "В диалоге я держу {style} тон.",
    "Люди говорят, что у меня {style} подход к знакомствам.",
    "Обычно передаю {style} атмосферу в общении.",
]

OPENING_TEMPLATES = [
    "Мне {age}, живу в {city}.",
    "Сейчас мне {age} лет, мой город {city}.",
    "Я из города {city}, мне {age}.",
    "{city} - мой дом, мне {age} лет.",
    "Привет, я из {city}, возраст {age}.",
    "По паспорту {age}, живу в {city}.",
]

NEGATION_TEMPLATES = [
    "Без суеты и без лишнего давления.",
    "Не люблю, когда все слишком резко и громко.",
    "Сильно ценю личное пространство и спокойный темп.",
    "Предпочитаю честный и понятный формат общения.",
]

def _pick_many(pool: List[str], min_n: int, max_n: int) -> List[str]:
    return random.sample(pool, random.randint(min_n, max_n))


def _make_nickname() -> str:
    return f"{random.choice(NICK_PARTS)}_{random.choice(NICK_PARTS)}_{random.randint(10, 99)}"


def _normalize_city(value: str) -> str:
    return value.lower().replace("-", " ").replace("  ", " ").strip()


def _pick_city_alias(city_key: str) -> str:
    options = CITY_ALIASES.get(city_key, [city_key.title()])
    return random.choice(options)


def _pick_with_custom(base_options: List[str], custom_options: List[str], custom_prob: float = 0.35) -> str:
    if random.random() < custom_prob:
        return random.choice(custom_options)
    return random.choice(base_options)


def _make_profile(user_id: int) -> Dict[str, object]:
    gender = random.choice(GENDERS)
    search_gender = "Женский" if gender == "Мужской" else "Мужской"
    city_key = _normalize_city(random.choice(CITIES))
    return {
        "user_id": user_id,
        "nickname": _make_nickname(),
        "gender": gender,
        "search_gender": search_gender,
        "city": _pick_city_alias(city_key),
        "city_key": city_key,
        "age": random.randint(*AGE_RANGE),
        "lifestyle": _pick_with_custom(["домашний", "смешанный", "активный"], ["гибридный", "спонтанный", "размеренный", "ночной", "интенсивный"]),
        "communication": _pick_with_custom(["спокойный", "нейтральный", "эмоциональный"], ["ироничный", "прямолинейный", "вдумчивый", "тактичный", "юмористичный"]),
        "values": _pick_with_custom(["семья", "баланс", "карьера"], ["саморазвитие", "свобода", "стабильность", "честность", "гармония"]),
        "tempo": _pick_with_custom(["медленный", "средний", "быстрый"], ["волнообразный", "проектный", "адаптивный", "динамичный"]),
        "hobbies": _pick_many(HOBBIES, 2, 4),
        "goal": random.choice(GOALS),
        "style": random.choice(["теплый", "спокойный", "динамичный", "осознанный", "легкий", "прямой", "доброжелательный"]),
        "arch": None,
    }


def _profile_from_arch(arch: str, user_id: int) -> Dict[str, object]:
    p = _make_profile(user_id)
    p["arch"] = arch
    meta = CATEGORY_META.get(arch, CATEGORY_META["generic"])
    if arch != "generic":
        if meta["tempo"] == "быстрый":
            p["tempo"] = _pick_with_custom(["быстрый", "средний"], ["динамичный", "проектный"], 0.15)
        elif meta["tempo"] == "медленный":
            p["tempo"] = _pick_with_custom(["медленный", "средний"], ["размеренный", "спокойный"], 0.15)
        if meta["openness"] == "высокая":
            p["style"] = random.choice(["легкий", "осознанный", "динамичный"])
        if arch in {"family_serious", "pet_lover"}:
            p["values"] = _pick_with_custom(["семья", "баланс"], ["стабильность", "гармония"], 0.25)
        if arch in {"career_hustle"}:
            p["values"] = _pick_with_custom(["карьера", "баланс"], ["саморазвитие", "свобода"], 0.2)
        if arch in {"casual_fwb"}:
            p["goal"] = random.choice(["хочется общения без лишнего давления", "смотрю на знакомства спокойно и без спешки", "интересно спокойное общение и встречи"])
    return p


def _apply_noise(text: str) -> str:
    if random.random() < 0.2:
        text = text.lower()
    if random.random() < 0.15:
        text = text.replace(".", "")
    if random.random() < 0.1:
        text = text.replace(",", "")
    if random.random() < 0.08:
        text = text.replace(" - ", " — ")
    if random.random() < 0.06:
        text = text.replace("я ", "Я ", 1)
    return " ".join(text.split())


def _text_from_profile(p: Dict[str, object]) -> str:
    hobbies = ", ".join(p["hobbies"])
    parts = [
        random.choice(OPENING_TEMPLATES).format(age=p["age"], city=p["city"]),
        random.choice(PERSONALITY_TEMPLATES).format(communication=p["communication"], lifestyle=p["lifestyle"]),
        random.choice(VALUES_TEMPLATES).format(values=p["values"]),
        random.choice(HOBBY_TEMPLATES).format(hobbies=hobbies),
        random.choice(STYLE_TEMPLATES).format(style=p["style"]),
        random.choice([f"Мой фокус - {p['goal']}.", f"Если про знакомства, то {p['goal']}.", f"Сейчас мой фокус - {p['goal']}."]),
    ]
    if random.random() < 0.35:
        parts.append(random.choice(NEGATION_TEMPLATES))
    if p.get("arch") in {"homebody_geek", "pet_lover", "family_serious"} and random.random() < 0.4:
        parts.append("Мне важны уют, спокойствие и нормальный темп.")
    if p.get("arch") in {"party_social", "travel_nomad", "active_extreme"} and random.random() < 0.4:
        parts.append("Люблю, когда в жизни есть движение и новые впечатления.")
    return _apply_noise(" ".join(parts))


def _mutate_similar(base: Dict[str, object], user_id: int) -> Dict[str, object]:
    p = dict(base)
    p["user_id"] = user_id
    p["nickname"] = _make_nickname()
    p["age"] = max(18, min(42, int(base["age"]) + random.randint(-3, 3)))
    p["city"] = base["city"] if random.random() < 0.7 else random.choice(CITIES)
    p["city_key"] = _normalize_city(p["city"])
    pool = list(dict.fromkeys(list(base["hobbies"]) + _pick_many(HOBBIES, 1, 2)))
    p["hobbies"] = random.sample(pool, min(len(pool), random.randint(2, 4)))
    if random.random() < 0.5:
        p["goal"] = base["goal"]
    if random.random() < 0.6:
        p["style"] = base["style"]
    if random.random() < 0.5:
        p["values"] = base["values"]
    if random.random() < 0.4:
        p["tempo"] = base["tempo"]
    return p


def _mutate_different(base: Dict[str, object], user_id: int) -> Dict[str, object]:
    p = _profile_from_arch(random.choice([k for k in CATEGORY_META.keys() if k not in {base.get("arch", "generic")}]), user_id)
    if p["city"] == base["city"]:
        p["city"] = random.choice([c for c in CITIES if c != base["city"]])
    if abs(int(p["age"]) - int(base["age"])) < 6:
        p["age"] = max(18, min(42, int(base["age"]) + random.choice([-11, -8, 8, 11])))
    p["city_key"] = _normalize_city(p["city"])
    return p


def _answers_json(profile: Dict[str, object]) -> str:
    answers = {
        "activity": profile["lifestyle"],
        "age": str(profile["age"]),
        "city": _normalize_city(str(profile["city"])),
        "communication": profile["communication"],
        "gender": profile["gender"],
        "nickname": profile["nickname"],
        "search_gender": profile["search_gender"],
        "tempo": profile["tempo"],
        "values": profile["values"],
    }
    return json.dumps(answers, ensure_ascii=False, sort_keys=True)


def _feedback_fields(label: str) -> Tuple[int, int, int]:
    if label == "positive":
        return 1, 1 if random.random() < 0.78 else 0, random.choice([4, 5])
    if label == "negative":
        return 0, 0, random.choice([1, 2])
    return (1 if random.random() < 0.55 else 0, 1 if random.random() < 0.25 else 0, random.choice([3, 4]))


def _compose_neutral_pair(left: Dict[str, object], right_id: int) -> Dict[str, object]:
    if random.random() < 0.5:
        right = _mutate_similar(left, right_id)
        # Keep some mismatch so neutral is not a duplicate of positive.
        if random.random() < 0.6:
            right["values"] = random.choice([v for v in ["семья", "баланс", "карьера", "свобода", "честность", "гармония"] if v != left["values"]])
        if random.random() < 0.5:
            right["tempo"] = random.choice([v for v in ["медленный", "средний", "быстрый", "адаптивный", "проектный", "динамичный"] if v != left["tempo"]])
    else:
        right = _profile_from_arch(random.choice(["generic", "family_serious", "homebody_geek", "pet_lover", "creative_art"]), right_id)
        if random.random() < 0.5:
            right["city"] = left["city"]
        if random.random() < 0.5:
            right["communication"] = left["communication"]
        if random.random() < 0.35:
            right["goal"] = left["goal"]
    return right


def _pair_for_label(label: str, left_id: int, right_id: int) -> Tuple[Dict[str, object], Dict[str, object]]:
    left_arch, right_arch, _ = random.choice(MATCH_RULES)
    left = _profile_from_arch(left_arch, left_id)

    if label == "positive":
        if random.random() < 0.7:
            right = _mutate_similar(left, right_id)
        else:
            right = _profile_from_arch(right_arch, right_id)
            if random.random() < 0.5:
                right["city"] = left["city"]
            if random.random() < 0.4:
                right["goal"] = left["goal"]
    elif label == "negative":
        right = _mutate_different(left, right_id)
    else:
        right = _compose_neutral_pair(left, right_id)

    return left, right


def _random_date(base_ts: datetime, feedback_id: int) -> str:
    delta_days = random.randint(0, 365)
    delta_minutes = random.randint(0, 24 * 60)
    dt = base_ts - timedelta(days=delta_days, minutes=delta_minutes, seconds=random.randint(0, 3600))
    jitter = timedelta(minutes=feedback_id % 7)
    return (dt + jitter).replace(microsecond=0).isoformat()


def generate_raw_dataset(output_csv: str, total_rows: int = 5000, seed: int = 42) -> None:
    random.seed(seed)
    labels = ["positive", "negative", "neutral"]
    per_label = total_rows // len(labels)
    remainder = total_rows % len(labels)

    rows: List[List[object]] = []
    seen = set()
    base_ts = datetime.now(timezone.utc)
    feedback_counter = 1
    user_counter = 100000

    for index, label in enumerate(labels):
        target = per_label + (1 if index < remainder else 0)
        created = 0
        guard = 0
        while created < target and guard < total_rows * 80:
            guard += 1
            left_id = user_counter
            right_id = user_counter + 1
            user_counter += 2

            left, right = _pair_for_label(label, left_id, right_id)
            text_left = _text_from_profile(left)
            text_right = _text_from_profile(right)
            key = (
                text_left,
                text_right,
                label,
                _normalize_city(str(left["city"])),
                _normalize_city(str(right["city"])),
            )
            if key in seen:
                continue
            seen.add(key)

            liked, meeting_agree, score = _feedback_fields(label)
            age_diff = abs(int(left["age"]) - int(right["age"]))
            rows.append(
                [
                    feedback_counter,
                    left_id,
                    right_id,
                    text_left,
                    text_right,
                    _answers_json(left),
                    _answers_json(right),
                    liked,
                    meeting_agree,
                    score,
                    label,
                    _random_date(base_ts, feedback_counter),
                    _normalize_city(str(left["city"])),
                    _normalize_city(str(right["city"])),
                    age_diff,
                ]
            )
            feedback_counter += 1
            created += 1

    random.shuffle(rows)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "feedback_id",
                "from_user_id",
                "to_user_id",
                "from_about",
                "to_about",
                "from_answers_json",
                "to_answers_json",
                "liked",
                "meeting_agree",
                "user_score",
                "label",
                "created_at",
                "from_city",
                "to_city",
                "age_diff",
            ]
        )
        writer.writerows(rows)

    print(f"Generated: {output_csv}")
    print(f"Rows: {len(rows)}")
    print(f"Per label: {per_label}")


def _write_rows(output_csv: str, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def split_dataset(
    input_csv: str,
    train_csv: str,
    val_csv: str,
    test_csv: str,
    seed: int = 42,
) -> Dict[str, int]:
    with open(input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise RuntimeError(f"No rows found in {input_csv}")

    labels = [row["label"] for row in rows]
    train_rows, temp_rows = train_test_split(
        rows,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    temp_labels = [row["label"] for row in temp_rows]
    val_rows, test_rows = train_test_split(
        temp_rows,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )

    _write_rows(train_csv, train_rows, fieldnames)
    _write_rows(val_csv, val_rows, fieldnames)
    _write_rows(test_csv, test_rows, fieldnames)

    return {
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
    }


def main() -> None:
    raw_path = "exports/nlp_dataset.csv"
    train_path = "exports/nlp_dataset_train.csv"
    val_path = "exports/nlp_dataset_val.csv"
    test_path = "exports/nlp_dataset_test.csv"
    generate_raw_dataset(output_csv=raw_path, total_rows=6000, seed=42)

    split_stats = split_dataset(
        input_csv=raw_path,
        train_csv=train_path,
        val_csv=val_path,
        test_csv=test_path,
        seed=42,
    )
    print(f"Built train dataset: {train_path}")
    print(f"Built validation dataset: {val_path}")
    print(f"Built test dataset: {test_path}")
    for key in sorted(split_stats.keys()):
        print(f"{key}: {split_stats[key]}")

    raw_file = Path(raw_path)
    if raw_file.exists():
        raw_file.unlink()
        print(f"Removed intermediate raw dataset: {raw_path}")


if __name__ == "__main__":
    main()
