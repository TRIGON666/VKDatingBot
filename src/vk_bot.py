from __future__ import annotations

import io
import logging
import random
import re
import os
import sys
from collections.abc import Mapping
from difflib import SequenceMatcher
from html import escape, unescape
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import vk_api
from vk_api.bot_longpoll import VkBotEventType, VkBotLongPoll
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.longpoll import VkEventType, VkLongPoll

from src.analytics import collect_feedback_stats
from src.database import Database, PhotoRecord, UserProfile
from src.matching import rank_matches
from src.questionnaire import QUESTIONS, calculate_questionnaire_compatibility
from src.psychology_questions import (
    ALL_QUESTIONS as PSYCHOLOGY_QUESTIONS,
    calculate_scores,
    format_profile_summary,
)

logger = logging.getLogger("vk_bot")


class VKCompatibilityBot:
    DAILY_LIKE_LIMIT = 50
    DAILY_DISLIKE_LIMIT = 200
    FEEDBACK_PAGE_SIZE = 10
    MAX_PHOTO_BYTES = 10 * 1024 * 1024
    ALLOWED_PHOTO_MIME_TYPES = {"image/jpeg", "image/png", "image/webp"}
    CITY_ALIASES = {
        "москва": {"москва", "мск", "msk", "moscow", "moskva", "default city"},
        "санкт петербург": {
            "санкт петербург",
            "петербург",
            "питер",
            "спб",
            "с пб",
            "санктпетербург",
            "spb",
            "saint petersburg",
            "st petersburg",
            "stpetersburg",
        },
        "новосибирск": {"новосибирск", "новосиб", "новосибрск", "нск", "nsk", "novosibirsk"},
        "екатеринбург": {"екатеринбург", "екб", "екат", "екатер", "ekb", "yekaterinburg", "sverdlovsk"},
        "нижний новгород": {"нижний новгород", "нижний", "ннов", "нн", "nn", "nnov", "novgorod"},
        "ростов на дону": {"ростов на дону", "ростов", "ростовдон", "rnd", "rostov", "rostov na donu"},
        "краснодар": {"краснодар", "крд", "krd", "krasnodar"},
        "самара": {"самара", "samara"},
        "казань": {"казань", "казан", "kazan", "kzn", "кзн"},
        "челябинск": {"челябинск", "chelyabinsk", "челяба", "челяб"},
        "уфа": {"уфа", "ufa"},
        "пермь": {"пермь", "perm"},
        "петрозаводск": {"петрозаводск", "птз", "ptz", "petrozavodsk", "petro"},
        "омск": {"омск", "omsk"},
        "воронеж": {"воронеж", "voronezh", "врн", "vrn"},
        "волгоград": {"волгоград", "volgograd", "влг"},
        "красноярск": {"красноярск", "krasnoyarsk", "крск", "krsk"},
        "сочи": {"сочи", "sochi"},
        "тюмень": {"тюмень", "tyumen", "тмн", "tmn"},
        "саратов": {"саратов", "saratov"},
        "тольятти": {"тольятти", "togliatti", "тлт", "tlt"},
        "ижевск": {"ижевск", "izhevsk", "иж"},
        "барнаул": {"барнаул", "barnaul"},
        "ульяновск": {"ульяновск", "ulyanovsk", "улн"},
        "иркутск": {"иркутск", "irkutsk", "ирк"},
        "хабаровск": {"хабаровск", "khabarovsk", "хаб"},
        "ярославль": {"ярославль", "yaroslavl", "яр"},
        "владивосток": {"владивосток", "vladivostok", "влд"},
        "махачкала": {"махачкала", "махач", "makhachkala"},
        "томск": {"томск", "tomsk"},
        "оренбург": {"оренбург", "orenburg", "орен"},
        "кемерово": {"кемерово", "kemerovo", "кем"},
        "новокузнецк": {"новокузнецк", "novokuznetsk", "нк"},
        "рязань": {"рязань", "ryazan", "рзн"},
        "астрахань": {"астрахань", "astrakhan"},
        "пенза": {"пенза", "penza"},
        "липецк": {"липецк", "lipetsk"},
        "киров": {"киров", "kirov"},
        "чебоксары": {"чебоксары", "cheboksary", "чебы"},
        "калининград": {"калининград", "kaliningrad", "кениг", "kenig"},
        "тула": {"тула", "tula"},
        "курск": {"курск", "kursk"},
        "севастополь": {"севастополь", "sevastopol", "севас"},
        "ставрополь": {"ставрополь", "stavropol"},
        "белгород": {"белгород", "belgorod"},
        "архангельск": {"архангельск", "arkhangelsk"},
        "смоленск": {"смоленск", "smolensk"},
        "владимир": {"владимир", "vladimir"},
        "саранск": {"саранск", "saransk"},
        "брянск": {"брянск", "bryansk"},
        "иваново": {"иваново", "ivanovo"},
        "сургут": {"сургут", "surgut"},
        "якутск": {"якутск", "yakutsk"},
    }
    def _log(self, message: str, *args: object) -> None:
        logger.info(message, *args)
    def _detect_group_id(self) -> Optional[int]:
        try:
            groups = self.vk.groups.getById()
            if groups:
                group = self._dict_like(groups[0])
                return int(group["id"])
        except Exception as e:
            self._log("Could not auto-detect VK_GROUP_ID: %s", e)
        return None

    # Инициализирует объект и сохраняет нужные зависимости.
    def __init__(
        self,
        token: str,
        db: Database,
        admin_ids: Optional[Set[int]] = None,
        group_id: Optional[int] = None,
    ) -> None:
        try:
            self.vk_session = vk_api.VkApi(token=token, api_version="5.131")
        except TypeError:
            self.vk_session = vk_api.VkApi(token=token)
        self.vk = self.vk_session.get_api()
        self.group_id = group_id or self._detect_group_id()
        self.bot_longpoll = VkBotLongPoll(self.vk_session, self.group_id) if self.group_id else None
        self.longpoll = None if self.bot_longpoll else VkLongPoll(self.vk_session)
        self.db = db
        self.admin_ids = admin_ids or set()
        self.user_states: Dict[int, Dict[str, object]] = {}
        self._nlp_metrics_tracker = None
        self._city_norm_cache: Dict[str, str] = {}
        self.city_alias_to_canonical: Dict[str, str] = {}
        for canonical, aliases in self.CITY_ALIASES.items():
            canon_norm = self._normalize_city_raw(canonical)
            self.city_alias_to_canonical[canon_norm] = canonical
            for alias in aliases:
                self.city_alias_to_canonical[self._normalize_city_raw(alias)] = canonical
        self._city_alias_items = list(self.city_alias_to_canonical.items())
        self._log(
            "VK bot initialized. Admins configured: %s. LongPoll mode: %s",
            len(self.admin_ids),
            "bot" if self.bot_longpoll else "legacy",
        )
    def _is_admin(self, user_id: int) -> bool:
        return user_id in self.admin_ids
    def _require_admin(self, user_id: int) -> bool:
        if self._is_admin(user_id):
            return True
        self._log("Denied admin command for user=%s", user_id)
        self.send_message(user_id, "Команда доступна только администратору.")
        return False
    def _track_nlp_interaction(
        self,
        user_id: int,
        candidate_id: int,
        action: str,
        state: Dict[str, object],
        idx: int,
    ) -> None:
        """Логировать пользовательское действие для обучения и метрик NLP."""
        try:
            from src.nlp_data_collector import log_interaction_for_nlp

            log_interaction_for_nlp(self.db, user_id, candidate_id, action)
        except Exception as e:
            self._log(f"Ошибка логирования данных NLP: user={user_id} candidate={candidate_id}: {e}")

        try:
            scores = state.get("candidate_scores", [])
            predicted_score = 0.0
            if isinstance(scores, list) and 0 <= idx < len(scores):
                predicted_score = float(scores[idx])

            if predicted_score >= 60:
                predicted_class = "positive"
            elif predicted_score <= 40:
                predicted_class = "negative"
            else:
                predicted_class = "neutral"

            self._log(
                "реакция_модели viewer=%s candidate=%s predicted_score=%.2f predicted_class=%s actual_action=%s",
                user_id,
                candidate_id,
                predicted_score,
                predicted_class,
                action,
            )

            if self._nlp_metrics_tracker is None:
                from src.nlp_metrics import NLPMetricsTracker

                self._nlp_metrics_tracker = NLPMetricsTracker()

            if self._nlp_metrics_tracker is not None:
                self._nlp_metrics_tracker.log_prediction(
                    viewer_id=user_id,
                    viewed_id=candidate_id,
                    predicted_score=predicted_score,
                    actual_action=action,
                    model_version="combined_v1",
                )
        except Exception as e:
            self._log(f"Ошибка логирования метрик NLP: user={user_id} candidate={candidate_id}: {e}")

    # Отправляет сообщение пользователю VK с клавиатурой или вложением.
    def send_message(
        self,
        user_id: int,
        message: str,
        keyboard: Optional[VkKeyboard] = None,
        attachment: Optional[str] = None,
    ) -> None:
        try:
            payload = {"user_id": user_id, "random_id": random.randint(1, 2**31 - 1), "message": message}
            if keyboard:
                payload["keyboard"] = keyboard.get_keyboard()
            if attachment:
                payload["attachment"] = attachment
            self.vk.messages.send(**payload)
        except Exception as e:
            self._log(f"ОШИБКА отправки сообщения user={user_id}: {str(e)}")
    def _get_nick(self, profile: Optional[UserProfile], fallback: str = "Пользователь") -> str:
        if not profile:
            return fallback
        nick = str(profile.questionnaire.get("nickname", "")).strip()
        return nick if nick else fallback

    # Нормализует или сопоставляет название города.
    def _normalize_city_raw(self, city: str) -> str:
        value = city.lower().replace("ё", "е").strip()
        value = re.sub(r"[.,\-_/]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        value = re.sub(r"\b(г|гор|город|обл|область|край|респ|республика|район|р н|рн)\b", " ", value)
        return re.sub(r"\s+", " ", value).strip()

    # Нормализует или сопоставляет название города.
    def _normalize_city(self, city: str) -> str:
        cached = self._city_norm_cache.get(city)
        if cached is not None:
            return cached

        value = self._normalize_city_raw(city)
        if not value:
            self._city_norm_cache[city] = ""
            return ""

        direct = self.city_alias_to_canonical.get(value)
        if direct:
            self._city_norm_cache[city] = direct
            return direct

        # Поиск алиасов в строках типа "санкт петербург центр" или "москва ювао".
        for alias, canonical in self._city_alias_items:
            if len(alias) >= 4 and (value in alias or alias in value):
                self._city_norm_cache[city] = canonical
                return canonical

        # Нечёткое сравнение ловит опечатки и близкие варианты названий.
        min_ratio = 0.84 if len(value) >= 6 else 0.9
        best_alias = ""
        best_ratio = 0.0
        for alias, _ in self._city_alias_items:
            ratio = SequenceMatcher(None, value, alias).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_alias = alias
        if best_alias and best_ratio >= min_ratio:
            result = self.city_alias_to_canonical[best_alias]
            self._city_norm_cache[city] = result
            return result

        self._city_norm_cache[city] = value
        return value

    # Создает клавиатуру VK из описания строк и кнопок.
    @staticmethod
    def _keyboard(rows: List[List[Tuple[str, str]]]) -> VkKeyboard:
        kb = VkKeyboard(one_time=False)
        for row_idx, row in enumerate(rows):
            if row_idx:
                kb.add_line()
            for label, color in row:
                kb.add_button(label, color=color)
        return kb
    def _save_create_state(self, user_id: int, state: Dict[str, Any]) -> None:
        self.user_states[user_id] = state
        self.db.save_draft(user_id, state)
    def _continue_create_step(self, user_id: int, state: Dict[str, Any]) -> bool:
        self._save_create_state(user_id, state)
        self._next_create_step(user_id)
        return True
    def _start_keyboard(self) -> VkKeyboard:
        return self._keyboard([[("Начать", VkKeyboardColor.POSITIVE)]])
    def _create_keyboard(self) -> VkKeyboard:
        return self._keyboard([[("Создать анкету", VkKeyboardColor.PRIMARY)]])

    # Работает с анкетой или профилем пользователя.
    def _profile_keyboard(self) -> VkKeyboard:
        return self._keyboard([
            [("Показать свою анкету", VkKeyboardColor.SECONDARY)],
            [("Смотреть анкеты", VkKeyboardColor.POSITIVE), ("Кто лайкнул", VkKeyboardColor.PRIMARY)],
            [("Мэтчи", VkKeyboardColor.POSITIVE), ("Оставить отзыв", VkKeyboardColor.SECONDARY)],
            [("Перезаполнить анкету", VkKeyboardColor.SECONDARY)],
        ])
    def _browse_keyboard(self) -> VkKeyboard:
        return self._keyboard([
            [("❤", VkKeyboardColor.POSITIVE), ("👎", VkKeyboardColor.NEGATIVE)],
            [("⚠ Жалоба", VkKeyboardColor.SECONDARY), ("🚫 Блок", VkKeyboardColor.SECONDARY)],
            [("Стоп", VkKeyboardColor.SECONDARY)],
        ])
    def _cancel_keyboard(self) -> VkKeyboard:
        return self._keyboard([[("Отмена", VkKeyboardColor.NEGATIVE)]])
    def _gender_keyboard(self) -> VkKeyboard:
        return self._keyboard([
            [("Мужской", VkKeyboardColor.PRIMARY), ("Женский", VkKeyboardColor.PRIMARY)],
            [("Отмена", VkKeyboardColor.NEGATIVE)],
        ])
    @staticmethod
    def _opposite_gender(gender: object) -> str:
        value = str(gender).strip().lower()
        if value == "мужской":
            return "Женский"
        if value == "женский":
            return "Мужской"
        return ""

    # Работает с вопросами анкеты или теста.
    def _question_keyboard(self, question_idx: int) -> VkKeyboard:
        rows: List[List[Tuple[str, str]]] = []
        row: List[Tuple[str, str]] = []
        for option in QUESTIONS[question_idx].options:
            row.append((option, VkKeyboardColor.PRIMARY))
            if len(row) == 2:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        rows.append([("Отмена", VkKeyboardColor.NEGATIVE)])
        return self._keyboard(rows)

    # Работает с фотографиями анкеты.
    def _photo_keyboard(self) -> VkKeyboard:
        return self._keyboard([[("Готово", VkKeyboardColor.POSITIVE), ("Отмена", VkKeyboardColor.NEGATIVE)]])
    def _yes_no_keyboard(self) -> VkKeyboard:
        return self._keyboard([[("Да", VkKeyboardColor.POSITIVE), ("Нет", VkKeyboardColor.NEGATIVE)]])
    def _rating_keyboard(self) -> VkKeyboard:
        return self._keyboard([
            [
                ("1", VkKeyboardColor.NEGATIVE),
                ("2", VkKeyboardColor.SECONDARY),
                ("3", VkKeyboardColor.SECONDARY),
                ("4", VkKeyboardColor.SECONDARY),
                ("5", VkKeyboardColor.POSITIVE),
            ],
            [("Отмена", VkKeyboardColor.NEGATIVE)],
        ])
    def _psychology_keyboard(self) -> VkKeyboard:
        """Клавиатура психологического опроса по шкале 1-5."""
        kb = VkKeyboard(one_time=False)
        kb.add_button("1 (не согласен)", color=VkKeyboardColor.NEGATIVE)
        kb.add_button("2", color=VkKeyboardColor.SECONDARY)
        kb.add_line()
        kb.add_button("3", color=VkKeyboardColor.SECONDARY)
        kb.add_line()
        kb.add_button("4", color=VkKeyboardColor.SECONDARY)
        kb.add_button("5 (согласен)", color=VkKeyboardColor.POSITIVE)
        kb.add_line()
        kb.add_button("Отмена", color=VkKeyboardColor.NEGATIVE)
        return kb

    # Работает с фотографиями анкеты.
    def _normalize_photo_attachment(self, raw: str) -> Optional[str]:
        value = str(raw).strip()
        if not value:
            return None
        if value.startswith("photo"):
            return value
        if "_" in value:
            return f"photo{value}"
        return None
    @staticmethod
    def _dict_like(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, Mapping):
            return dict(value)
        items = getattr(value, "items", None)
        if callable(items):
            try:
                return dict(items())
            except (TypeError, ValueError):
                pass
        data = getattr(value, "__dict__", None)
        if isinstance(data, dict):
            return data
        return {}

    # Достает поле из VK-события в едином формате.
    def _event_message(self, event) -> Dict[str, Any]:
        for attr in ("message",):
            value = self._dict_like(getattr(event, attr, None))
            if value:
                return value
        for container_attr in ("object", "obj"):
            container = getattr(event, container_attr, None)
            container_data = self._dict_like(container)
            if container_data:
                message = self._dict_like(container_data.get("message"))
                if message:
                    return message
                return container_data
            message = self._dict_like(getattr(container, "message", None))
            if message:
                return message
        return {}

    # Достает поле из VK-события в едином формате.
    def _event_user_id(self, event) -> int:
        value = getattr(event, "user_id", None)
        if value is not None:
            return int(value)
        message = self._event_message(event)
        return int(message.get("from_id") or message.get("user_id") or 0)

    # Достает поле из VK-события в едином формате.
    def _event_peer_id(self, event) -> int:
        value = getattr(event, "peer_id", None)
        if value is not None:
            return int(value)
        message = self._event_message(event)
        return int(message.get("peer_id") or self._event_user_id(event))

    # Достает поле из VK-события в едином формате.
    def _event_text(self, event) -> str:
        value = getattr(event, "text", None)
        if value is not None:
            return str(value)
        return str(self._event_message(event).get("text") or "")

    # Работает с фотографиями анкеты.
    def _extract_photo_attachments(self, event) -> List[str]:
        attachments: List[str] = []
        message = self._event_message(event)
        for item in message.get("attachments", []) if isinstance(message.get("attachments"), list) else []:
            item = self._dict_like(item)
            if item.get("type") != "photo":
                continue
            photo_id = self._photo_attachment_id(item.get("photo", {}))
            if photo_id:
                attachments.append(photo_id)
        if hasattr(event, "attachments"):
            old = event.attachments
            idx = 1
            while True:
                key_type = f"attach{idx}_type"
                key_val = f"attach{idx}"
                key_access = f"attach{idx}_access_key"
                if key_type not in old:
                    break
                if old.get(key_type) == "photo" and key_val in old:
                    base = str(old[key_val]).strip()
                    if not base.startswith("photo"):
                        base = f"photo{base}"
                    access_key = old.get(key_access)
                    if access_key and len(base.split("_")) < 3:
                        base = f"{base}_{access_key}"
                    attachments.append(base)
                idx += 1
        return attachments
    def _message_items_by_id(self, peer_id: int, message_id: Optional[int]) -> List[Dict[str, Any]]:
        if not message_id:
            return []
        for method_name, kwargs in (
            ("getById", {"message_ids": message_id}),
            ("getByConversationMessageId", {"peer_id": peer_id, "conversation_message_ids": message_id}),
        ):
            try:
                response = self._dict_like(getattr(self.vk.messages, method_name)(**kwargs))
                items = response.get("items", [])
                if items:
                    return items
            except Exception as e:
                self._log("VK %s did not return message %s for peer=%s: %s", method_name, message_id, peer_id, e)
        return []

    # Работает с фотографиями анкеты.
    def _extract_photo_attachments_by_message_id(self, peer_id: int, message_id: Optional[int]) -> List[str]:
        try:
            items = self._message_items_by_id(peer_id, message_id)
            if not items:
                return []
            result: List[str] = []
            for item in items[0].get("attachments", []):
                item = self._dict_like(item)
                if item.get("type") != "photo":
                    continue
                photo_id = self._photo_attachment_id(item.get("photo", {}))
                if photo_id:
                    result.append(photo_id)
            return result
        except (ValueError, KeyError, IndexError):
            return []
        except Exception as e:
            self._log(f"Ошибка извлечения вложений с фото: {e}")
            return []

    # Достает поле из VK-события в едином формате.
    def _event_message_id(self, event) -> Optional[int]:
        for attr in ("message_id", "id"):
            value = getattr(event, attr, None)
            if value:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

        message = self._event_message(event)
        for key in ("id", "conversation_message_id"):
            value = message.get(key)
            if value:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass

        raw = getattr(event, "raw", None)
        if isinstance(raw, (list, tuple)) and len(raw) > 1:
            try:
                return int(raw[1])
            except (TypeError, ValueError):
                return None
        if isinstance(raw, dict):
            candidates = [
                raw.get("message_id"),
                raw.get("id"),
                raw.get("object", {}).get("message", {}).get("id") if isinstance(raw.get("object"), dict) else None,
            ]
            for value in candidates:
                if value:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        pass
        return None

    # Работает с фотографиями анкеты.
    def _best_photo_url(self, photo: Dict[str, Any]) -> Optional[str]:
        photo = self._dict_like(photo)
        sizes = photo.get("sizes", [])
        if sizes:
            normalized_sizes = [self._dict_like(size) for size in sizes]
            best = max(normalized_sizes, key=lambda s: int(s.get("width", 0)) * int(s.get("height", 0)))
            url = best.get("url")
            if url:
                return url

        for key in ("src_xxxbig", "src_xxbig", "src_xbig", "src_big", "src", "photo_2560", "photo_1280", "photo_807", "photo_604"):
            url = photo.get(key)
            if url:
                return str(url)
        return None

    # Работает с фотографиями анкеты.
    def _photo_attachment_id(self, photo: Any, include_access_key: bool = True) -> Optional[str]:
        photo = self._dict_like(photo)
        owner_id = photo.get("owner_id")
        photo_id = photo.get("id") or photo.get("pid")
        if owner_id is None or photo_id is None:
            return None
        access_key = photo.get("access_key") if include_access_key else None
        return f"photo{owner_id}_{photo_id}_{access_key}" if access_key else f"photo{owner_id}_{photo_id}"

    # Работает с фотографиями анкеты.
    def _photo_attachment_key(self, photo: Dict[str, Any]) -> Optional[str]:
        return self._photo_attachment_id(photo, include_access_key=False)
    def _detect_image_mime(self, data: bytes, content_type: str) -> Optional[str]:
        mime = content_type.split(";", 1)[0].strip().lower()
        if mime == "image/jpg":
            return "image/jpeg"
        if mime in self.ALLOWED_PHOTO_MIME_TYPES:
            return mime
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        return None

    # Работает с фотографиями анкеты.
    def _download_photo_url(self, photo_url: str, log_context: str = "photo") -> Optional[Tuple[bytes, str, str]]:
        photo_url = unescape(str(photo_url).strip())
        if not photo_url:
            return None
        try:
            response = requests.get(
                photo_url,
                headers={"User-Agent": "VKDatingBot/1.0"},
                timeout=20,
            )
        except requests.RequestException as e:
            self._log("Failed to download %s from VK CDN: %s", log_context, e)
            return None

        try:
            if response.status_code != 200:
                self._log("Failed to download %s from VK CDN: HTTP %s", log_context, response.status_code)
                return None

            data = response.content[: self.MAX_PHOTO_BYTES + 1]
            if len(data) > self.MAX_PHOTO_BYTES:
                self._log("Skipped %s larger than %s bytes", log_context, self.MAX_PHOTO_BYTES)
                return None
            content_type = response.headers.get("content-type", "image/jpeg")
            mime_type = self._detect_image_mime(data, content_type)
            if not mime_type:
                self._log(
                    "Skipped %s with unsupported content type %s and magic bytes %s",
                    log_context,
                    content_type,
                    data[:12].hex(),
                )
                return None
            ext = "jpg"
            if mime_type == "image/png":
                ext = "png"
            elif mime_type == "image/webp":
                ext = "webp"
            return data, mime_type, f"photo.{ext}"
        except Exception as e:
            self._log("Failed to process %s from VK CDN: %s", log_context, e)
            return None
        finally:
            response.close()

    # Работает с фотографиями анкеты.
    def _download_photo_blob_from_attachment(
        self,
        attachment: Dict[str, Any],
        log_context: str,
    ) -> Optional[Tuple[bytes, str, str]]:
        attachment = self._dict_like(attachment)
        if attachment.get("type") != "photo":
            return None
        photo_url = self._best_photo_url(attachment.get("photo", {}))
        if not photo_url:
            return None
        return self._download_photo_url(photo_url, log_context)

    # Работает с фотографиями анкеты.
    def _download_photo_blobs_from_attachments(
        self,
        attachments: List[Any],
        log_context: str,
    ) -> List[Tuple[bytes, str, str]]:
        blobs: List[Tuple[bytes, str, str]] = []
        for attachment in attachments:
            blob = self._download_photo_blob_from_attachment(self._dict_like(attachment), log_context)
            if blob:
                blobs.append(blob)
        return blobs

    # Работает с фотографиями анкеты.
    def _download_photo_blobs_from_event(self, event) -> List[Tuple[bytes, str, str]]:
        message = self._event_message(event)
        items = message.get("attachments", [])
        if not isinstance(items, list):
            return []
        return self._download_photo_blobs_from_attachments(items, "photo from bot longpoll event")

    # Работает с фотографиями анкеты.
    def _photo_event_debug(self, event) -> List[Dict[str, Any]]:
        message = self._event_message(event)
        items = message.get("attachments", [])
        if not isinstance(items, list):
            return []
        result: List[Dict[str, Any]] = []
        for item in items:
            item = self._dict_like(item)
            photo = self._dict_like(item.get("photo", {}))
            sizes = [self._dict_like(size) for size in photo.get("sizes", [])] if isinstance(photo.get("sizes"), list) else []
            result.append(
                {
                    "type": item.get("type"),
                    "item_keys": sorted(item.keys()),
                    "photo_keys": sorted(photo.keys()),
                    "sizes_count": len(sizes),
                    "first_size_keys": sorted(sizes[0].keys()) if sizes else [],
                }
            )
        return result

    # Работает с фотографиями анкеты.
    def _download_photo_blobs_from_history(
        self,
        peer_id: int,
        expected_photos: Optional[List[str]] = None,
    ) -> List[Tuple[bytes, str, str]]:
        expected = {p.rsplit("_", 1)[0] if p.count("_") > 1 else p for p in (expected_photos or [])}
        try:
            response = self._dict_like(
                self.vk.messages.getHistoryAttachments(
                    peer_id=peer_id,
                    media_type="photo",
                    count=20,
                    preserve_order=1,
                )
            )
            result: List[Tuple[bytes, str, str]] = []
            for item in response.get("items", []):
                item = self._dict_like(item)
                attachment = self._dict_like(item.get("attachment", item))
                photo = self._dict_like(attachment.get("photo", {}))
                photo_key = self._photo_attachment_key(photo)
                if expected and photo_key not in expected:
                    continue
                blob = self._download_photo_blob_from_attachment(attachment, "photo from history attachments")
                if blob:
                    result.append(blob)
            return result
        except Exception as e:
            self._log("VK getHistoryAttachments did not return photo urls for peer=%s: %s", peer_id, e)
            return []

    # Работает с фотографиями анкеты.
    def _download_photo_blobs_by_message_id(
        self,
        peer_id: int,
        message_id: Optional[int],
        expected_photos: Optional[List[str]] = None,
    ) -> List[Tuple[bytes, str, str]]:
        try:
            items = self._message_items_by_id(peer_id, message_id)
            if not items:
                return self._download_photo_blobs_from_history(peer_id, expected_photos)
            message = self._dict_like(items[0])
            attachments = message.get("attachments", [])
            result = self._download_photo_blobs_from_attachments(
                attachments if isinstance(attachments, list) else [],
                f"photo from message {message_id}",
            )
            if result:
                return result
            return self._download_photo_blobs_from_history(peer_id, expected_photos)
        except (KeyError, IndexError):
            return []
        except Exception as e:
            self._log(f"Ошибка загрузки фото из сообщения {message_id}: {e}")
            return []

    # Работает с фотографиями анкеты.
    def _upload_photo_record_to_messages(self, user_id: int, photo: PhotoRecord) -> Optional[str]:
        try:
            photo_data = photo.get("photo_data")
            mime_type = photo.get("mime_type", "image/jpeg")
            if not isinstance(photo_data, (bytes, bytearray)) or not photo_data:
                self._log("Skipped empty photo upload for user=%s", user_id)
                return None
            if len(photo_data) > self.MAX_PHOTO_BYTES:
                self._log("Skipped photo upload larger than %s bytes for user=%s", self.MAX_PHOTO_BYTES, user_id)
                return None
            if mime_type not in self.ALLOWED_PHOTO_MIME_TYPES:
                self._log("Skipped photo upload with unsupported content type %s for user=%s", mime_type, user_id)
                return None
            upload_server = self.vk.photos.getMessagesUploadServer(peer_id=user_id)
            upload_url = upload_server.get("upload_url")
            if not upload_url:
                return None
            uploaded = requests.post(
                upload_url,
                files={
                    "photo": (
                        photo.get("filename", "photo.jpg"),
                        io.BytesIO(photo_data),
                        mime_type,
                    )
                },
                timeout=30,
            )
            try:
                if uploaded.status_code != 200:
                    self._log("VK message photo upload failed for user=%s: HTTP %s", user_id, uploaded.status_code)
                    return None
                payload = uploaded.json()
            finally:
                uploaded.close()
            saved = self.vk.photos.saveMessagesPhoto(
                photo=payload.get("photo"),
                server=payload.get("server"),
                hash=payload.get("hash"),
            )
            if not saved:
                self._log("VK saveMessagesPhoto returned empty response for user=%s", user_id)
                return None
            p = saved[0]
            owner_id = p.get("owner_id")
            photo_id = p.get("id")
            access_key = p.get("access_key")
            if owner_id is None or photo_id is None:
                self._log("VK saveMessagesPhoto response has no owner/id for user=%s: %s", user_id, p)
                return None
            return f"photo{owner_id}_{photo_id}_{access_key}" if access_key else f"photo{owner_id}_{photo_id}"
        except Exception as e:
            self._log("Failed to upload saved photo to VK messages for user=%s: %s", user_id, e)
            return None

    # Загружает скачиваемый файл в сообщения VK как документ.
    def _upload_document_to_messages(
        self,
        user_id: int,
        file_data: bytes,
        filename: str,
        mime_type: str = "application/octet-stream",
        title: Optional[str] = None,
    ) -> Optional[str]:
        try:
            if not isinstance(file_data, (bytes, bytearray)) or not file_data:
                self._log("Skipped empty document upload for user=%s", user_id)
                return None
            upload_server = self.vk.docs.getMessagesUploadServer(peer_id=user_id, type="doc")
            upload_url = upload_server.get("upload_url")
            if not upload_url:
                return None
            uploaded = requests.post(
                upload_url,
                files={"file": (filename, io.BytesIO(file_data), mime_type)},
                timeout=30,
            )
            try:
                if uploaded.status_code != 200:
                    self._log("VK document upload failed for user=%s: HTTP %s", user_id, uploaded.status_code)
                    return None
                payload = uploaded.json()
            finally:
                uploaded.close()
            file_token = payload.get("file")
            if not file_token:
                self._log("VK document upload returned no file token for user=%s: %s", user_id, payload)
                return None

            saved = self.vk.docs.save(file=file_token, title=title or filename)
            doc = None
            if isinstance(saved, dict):
                doc = saved.get("doc")
                if doc is None and isinstance(saved.get("docs"), list) and saved["docs"]:
                    doc = saved["docs"][0]
            elif isinstance(saved, list) and saved:
                first = saved[0]
                doc = first.get("doc") if isinstance(first, dict) and "doc" in first else first
            if not isinstance(doc, dict):
                self._log("VK docs.save returned unexpected payload for user=%s: %s", user_id, saved)
                return None
            owner_id = doc.get("owner_id")
            doc_id = doc.get("id")
            access_key = doc.get("access_key")
            if owner_id is None or doc_id is None:
                self._log("VK saved document has no owner/id for user=%s: %s", user_id, doc)
                return None
            return f"doc{owner_id}_{doc_id}_{access_key}" if access_key else f"doc{owner_id}_{doc_id}"
        except Exception as e:
            self._log("Failed to upload document to VK messages for user=%s: %s", user_id, e)
            return None

    # Работает с анкетой или профилем пользователя.
    def _is_profile_complete(self, profile: Optional[UserProfile]) -> bool:
        if not profile:
            return False
        required = {"age", "gender", "nickname"}
        if not required.issubset(set(profile.questionnaire.keys())):
            return False
        target_gender = str(profile.questionnaire.get("search_gender", "")).strip()
        if not target_gender:
            target_gender = self._opposite_gender(profile.questionnaire.get("gender", ""))
            if not target_gender:
                return False
            profile.questionnaire["search_gender"] = target_gender
        for q in QUESTIONS:
            if q.key not in profile.questionnaire:
                return False
        if not profile.photos:
            return False
        return True

    # Работает с анкетой или профилем пользователя.
    def _start_create_profile(self, user_id: int, reset: bool) -> None:
        self.db.register_user(user_id)
        if reset:
            self.db.clear_user_profile(user_id)
        self.user_states[user_id] = {
            "mode": "create_profile",
            "step": "nickname",
            "answers": {},
            "about_text": "",
            "psychology_answers": {},
            "psychology_question_idx": 0,
            "psychology_intro_shown": False,
        }
        self.db.save_draft(user_id, self.user_states[user_id])
        total_steps = 6 + len(QUESTIONS) + len(PSYCHOLOGY_QUESTIONS)
        self._log(f"user={user_id} started profile flow reset={reset}")
        self.send_message(
            user_id,
            "✨ Начинаем заполнять анкету!\n"
            "Шаг 1/{}: введите никнейм или имя.".format(total_steps),
            keyboard=self._cancel_keyboard(),
        )
    def _next_create_step(self, user_id: int) -> None:
        state = self.user_states.get(user_id, {})
        step = state.get("step")
        total_steps = 6 + len(QUESTIONS) + len(PSYCHOLOGY_QUESTIONS)
        
        if step == "nickname":
            self.send_message(
                user_id,
                "✨ Шаг 1/{}: введите никнейм или имя.".format(total_steps),
                keyboard=self._cancel_keyboard(),
            )
            return
        if step == "gender":
            self.send_message(
                user_id,
                "🧍 Шаг 2/{}: выберите ваш пол.".format(total_steps),
                keyboard=self._gender_keyboard(),
            )
            return
        if step == "age":
            self.send_message(
                user_id,
                "🎂 Шаг 3/{}: укажите возраст числом (от 12 до 99).".format(total_steps),
                keyboard=self._cancel_keyboard(),
            )
            return
        if step == "city":
            self.send_message(
                user_id,
                "🏙️ Шаг 4/{}: введите ваш город.".format(total_steps),
                keyboard=self._cancel_keyboard(),
            )
            return
        if step == "search_gender":
            answers = state.get("answers", {})
            if isinstance(answers, dict):
                answers["search_gender"] = self._opposite_gender(answers.get("gender", ""))
            state["step"] = "questions"
            state["question_idx"] = 0
            self._save_create_state(user_id, state)
            self._next_create_step(user_id)
            return
        if step == "questions":
            idx = int(state.get("question_idx", 0))
            if idx < 0 or idx >= len(QUESTIONS):
                state["step"] = "about"
                state.pop("question_idx", None)
                self.user_states[user_id] = state
                self.db.save_draft(user_id, state)
                self._next_create_step(user_id)
                return
            q = QUESTIONS[idx]
            self.send_message(
                user_id,
                f"🎯 Шаг {5 + idx}/{total_steps}: {q.text}",
                keyboard=self._question_keyboard(idx),
            )
            return
        if step == "about":
            self.send_message(
                user_id,
                "📝 Шаг {}/{}: расскажите о себе. Ваши увлечения, чем занимаетесь в свободное время, учитесь/работаете?".format(
                    5 + len(QUESTIONS), total_steps
                ),
                keyboard=self._cancel_keyboard(),
            )
            return
        if step == "photos":
            self.send_message(
                user_id,
                "📸 Шаг {}/{}: загрузите минимум 1 фото.\nПосле отправки нажмите 'Готово'.".format(
                    6 + len(QUESTIONS), total_steps
                ),
                keyboard=self._photo_keyboard(),
            )
            return
        if step == "psychology_questions":
            idx = int(state.get("psychology_question_idx", 0))
            if idx < 0:
                idx = 0
                state["psychology_question_idx"] = idx
                self.user_states[user_id] = state
                self.db.save_draft(user_id, state)
            if idx == 0 and not state.get("psychology_intro_shown"):
                self.send_message(
                    user_id,
                    "🧠 Начинаем психологический тест.\n"
                    "Перед каждым вопросом выберите число от 1 до 5:\n"
                    "1 — совсем не согласен\n"
                    "2 — скорее не согласен\n"
                    "3 — нейтрально\n"
                    "4 — скорее согласен\n"
                    "5 — полностью согласен",
                    keyboard=self._psychology_keyboard(),
                )
                state["psychology_intro_shown"] = True
                self.user_states[user_id] = state
                self.db.save_draft(user_id, state)
            if idx >= len(PSYCHOLOGY_QUESTIONS):
                self._finish_create_profile(user_id)
                return
            q = PSYCHOLOGY_QUESTIONS[idx]
            step_num = 7 + len(QUESTIONS) + idx
            self.send_message(
                user_id,
                f"💭 Шаг {step_num}/{total_steps}: {q['text']}\n\n(Ответьте числом 1-5, где 1 = совсем не согласен, 5 = полностью согласен)",
                keyboard=self._psychology_keyboard(),
            )

    # Работает с анкетой или профилем пользователя.
    def _finish_create_profile(self, user_id: int) -> None:
        state = self.user_states.get(user_id, {})
        answers = state.get("answers", {})
        if not isinstance(answers, dict):
            answers = {}
        target_gender = str(answers.get("search_gender", "")).strip()
        if not target_gender:
            opposite = self._opposite_gender(answers.get("gender", ""))
            if opposite:
                answers["search_gender"] = opposite
        about_text = state.get("about_text", "")
        if not isinstance(about_text, str):
            about_text = ""
        
        # Обработка психологических ответов
        psychology_answers = state.get("psychology_answers", {})
        if not isinstance(psychology_answers, dict):
            psychology_answers = {}
        
        # Обработка психологических средних
        psychology_scores = None
        if psychology_answers:
            psychology_scores = calculate_scores(psychology_answers)
        
        self.db.save_questionnaire(user_id, answers)
        self.db.save_text_profile(user_id, about_text)
        if psychology_answers:
            self.db.save_psychology_answers(user_id, psychology_answers, psychology_scores)
        
        self.db.clear_draft(user_id)
        self.db.log_event(user_id, "profile_complete")
        self.user_states.pop(user_id, None)
        
        # Построить сообщение целостности
        msg = "✅ Анкета заполнена и сохранена!\n\n"
        if psychology_scores:
            msg += format_profile_summary(psychology_scores)
            msg += "\n\n"
        msg += "Выберите действие в меню профиля."
        
        self.send_message(user_id, msg, keyboard=self._profile_keyboard())

    # Работает с анкетой или профилем пользователя.
    def _handle_create_profile(self, user_id: int, event) -> bool:
        state = self.user_states.get(user_id)
        if not state or state.get("mode") != "create_profile":
            return False
        text = self._event_text(event).strip()
        lowered = text.lower()
        if lowered == "отмена":
            self.user_states.pop(user_id, None)
            self.db.clear_draft(user_id)
            self._log(f"user={user_id} cancelled profile flow")
            self.send_message(user_id, "Заполнение анкеты отменено.", keyboard=self._create_keyboard())
            return True

        answers = state.get("answers", {})
        if not isinstance(answers, dict):
            answers = {}
            state["answers"] = answers

        step = state.get("step")
        if step == "nickname":
            nickname = text.strip()
            if not nickname or len(nickname) < 2 or len(nickname) > 32:
                self.send_message(
                    user_id,
                    "Ник должен быть длиной от 2 до 32 символов.",
                    keyboard=self._cancel_keyboard(),
                )
                return True
            answers["nickname"] = nickname
            state["step"] = "gender"
            return self._continue_create_step(user_id, state)

        if step == "gender":
            if text not in {"Мужской", "Женский"}:
                self.send_message(user_id, "Выберите ваш пол кнопкой ниже.", keyboard=self._gender_keyboard())
                return True
            answers["gender"] = text
            answers["search_gender"] = self._opposite_gender(text)
            state["step"] = "age"
            return self._continue_create_step(user_id, state)

        if step == "age":
            if not text.isdigit() or int(text) < 12 or int(text) > 99:
                self.send_message(
                    user_id,
                    "Возраст указан неверно. Введите целое число от 12 до 99.",
                    keyboard=self._cancel_keyboard(),
                )
                return True
            answers["age"] = text
            state["step"] = "city"
            return self._continue_create_step(user_id, state)

        if step == "city":
            city = text.strip()
            if len(city) < 2:
                self.send_message(user_id, "Город слишком короткий.", keyboard=self._cancel_keyboard())
                return True
            answers["city"] = self._normalize_city(city)
            state["step"] = "questions"
            state["question_idx"] = 0
            return self._continue_create_step(user_id, state)

        if step == "search_gender":
            answers["search_gender"] = self._opposite_gender(answers.get("gender", ""))
            state["step"] = "questions"
            state["question_idx"] = 0
            return self._continue_create_step(user_id, state)

        if step == "questions":
            idx = int(state.get("question_idx", 0))
            if idx < 0 or idx >= len(QUESTIONS):
                state["step"] = "about"
                state.pop("question_idx", None)
                return self._continue_create_step(user_id, state)

            q = QUESTIONS[idx]
            option_map = {str(opt).strip().lower(): opt for opt in q.options}
            selected = option_map.get(text.strip().lower())
            if selected is None:
                self.send_message(
                    user_id,
                    "Выберите один из готовых вариантов кнопкой ниже.",
                    keyboard=self._question_keyboard(idx),
                )
                return True

            answers[q.key] = selected
            idx += 1
            if idx >= len(QUESTIONS):
                state["step"] = "about"
                state.pop("question_idx", None)
            else:
                state["question_idx"] = idx
            return self._continue_create_step(user_id, state)

        if step == "psychology_questions":
            # Принять текст как плановые цифры и метки клавиатуры наподобие "5 (согласен)".
            answer_match = re.match(r"^\s*([1-5])", text)
            if not answer_match:
                self.send_message(
                    user_id,
                    "❌ Пожалуйста, введите число от 1 до 5.",
                    keyboard=self._psychology_keyboard(),
                )
                return True
            answer = int(answer_match.group(1))
            
            state = self.user_states.get(user_id, {})
            idx = int(state.get("psychology_question_idx", 0))
            
            if idx >= len(PSYCHOLOGY_QUESTIONS):
                self._finish_create_profile(user_id)
                return True
            
            q = PSYCHOLOGY_QUESTIONS[idx]
            q_id = q.get("id")
            if not q_id:
                idx += 1
                state["psychology_question_idx"] = idx
                return self._continue_create_step(user_id, state)
            psych_answers = state.get("psychology_answers", {})
            psych_answers[q_id] = answer
            
            idx += 1
            state["psychology_question_idx"] = idx
            state["psychology_answers"] = psych_answers
            
            if idx >= len(PSYCHOLOGY_QUESTIONS):
                self._save_create_state(user_id, state)
                self._finish_create_profile(user_id)
            else:
                self._continue_create_step(user_id, state)
            
            return True

        if step == "about":
            state["about_text"] = "" if text == "-" else text
            state["step"] = "photos"
            return self._continue_create_step(user_id, state)

        if step == "photos":
            if lowered == "готово":
                total = len(self.db.get_user_photos(user_id))
                if total < 1:
                    self.send_message(user_id, "Нужно загрузить хотя бы одно фото перед завершением.", keyboard=self._photo_keyboard())
                    return True
                # Переход к психологическому опросу.
                state = self.user_states.get(user_id, {})
                state["step"] = "psychology_questions"
                state["psychology_question_idx"] = 0
                state["psychology_answers"] = {}
                state["psychology_intro_shown"] = False
                return self._continue_create_step(user_id, state)
            message_id = self._event_message_id(event)
            peer_id = self._event_peer_id(event)
            self._log("photo upload event user=%s peer=%s message_id=%s", user_id, peer_id, message_id)
            photos = self._extract_photo_attachments(event)
            if not photos:
                photos = self._extract_photo_attachments_by_message_id(peer_id, message_id)
            photos = [p for p in (self._normalize_photo_attachment(v) for v in photos) if p]
            blobs = self._download_photo_blobs_from_event(event)
            if not blobs:
                blobs = self._download_photo_blobs_by_message_id(peer_id, message_id, photos)
            if blobs:
                ready = [{"photo_data": b[0], "mime_type": b[1], "filename": b[2]} for b in blobs]
                total = self.db.add_user_photos(user_id, ready)
                self.db.log_event(user_id, "photo_uploaded")
                self._log("user=%s uploaded photos count=%s total=%s", user_id, len(ready), total)
                self.send_message(
                    user_id,
                    f"Фото добавлены: +{len(ready)}. Сейчас в анкете: {total}.",
                    keyboard=self._photo_keyboard(),
                )
                return True
            if not photos:
                self.send_message(
                    user_id,
                    "Я не вижу фото во вложении. Отправьте фото или нажмите 'Готово'.",
                    keyboard=self._photo_keyboard(),
                )
                return True
            self._log(
                "Photo attachment ids found without downloadable message URLs: user=%s peer=%s message_id=%s photos=%s event_photo_debug=%s",
                user_id,
                peer_id,
                message_id,
                photos,
                self._photo_event_debug(event),
            )
            self.send_message(
                user_id,
                "Не удалось обработать это фото. Попробуйте отправить фото ещё раз.",
                keyboard=self._photo_keyboard(),
            )
            return True
        return False

    # Работает с анкетой или профилем пользователя.
    def handle_show_profile(self, user_id: int) -> None:
        profile = self.db.get_user_profile(user_id)
        if not self._is_profile_complete(profile):
            self.send_message(user_id, "Анкета не заполнена полностью. Нажмите 'Создать анкету'.", keyboard=self._create_keyboard())
            return
        assert profile is not None, f"Profile should be complete for user {user_id} but is None"
        q = profile.questionnaire
        own_nick = self._get_nick(profile)
        lines = [
            f"Ник: {own_nick}",
            "",
            f"Возраст: {q.get('age', '-')}",
            f"Пол: {q.get('gender', '-')}",
            f"Город: {q.get('city', '-').title()}",
            "",
        ]
        for item in QUESTIONS:
            lines.append(f"{item.text}: {q.get(item.key, '-')}")

        psych_scores = self.db.get_psychology_scores(user_id)
        if psych_scores:
            lines.extend(["", format_profile_summary(psych_scores)])

        lines.extend(["", f"О себе: {profile.about_text or '—'}"])
        attachments = [a for a in (self._upload_photo_record_to_messages(user_id, v) for v in profile.photos[:5]) if a]
        self.send_message(
            user_id,
            "\n".join(lines),
            keyboard=self._profile_keyboard(),
            attachment=",".join(attachments) if attachments else None,
        )

    # Работает с подбором и совместимостью анкет.
    def _format_match_card(
        self,
        viewer_user_id: int,
        profile: UserProfile,
        compatibility: float,
        score_details: Optional[Dict[str, Optional[float]]] = None,
    ) -> tuple[str, Optional[str]]:
        q = profile.questionnaire
        nick = self._get_nick(profile)
        def _fmt_component(value: Optional[float]) -> str:
            return f"{value:.1f}%" if value is not None else "н/д"

        lines = [
            f"Ник: {nick}",
            f"Совместимость: {compatibility}%",
            "",
            f"Возраст: {q.get('age', '-')}",
            f"Пол: {q.get('gender', '-')}",
            f"Город: {q.get('city', '-')}",
            "",
        ]
        for item in QUESTIONS:
            lines.append(f"{item.text}: {q.get(item.key, '-')}")
        if score_details:
            lines.extend(
                [
                    "",
                    "Разбор совместимости:",
                    f"- Анкета: {_fmt_component(score_details.get('questionnaire_score'))}",
                    f"- TF-IDF: {_fmt_component(score_details.get('tfidf_score'))}",
                    f"- NLP (калибр.): {_fmt_component(score_details.get('nlp_score'))}",
                    f"- Психотест: {_fmt_component(score_details.get('psychology_score'))}",
                ]
            )
        lines.extend(["", f"О себе: {profile.about_text or '—'}"])
        attachments = [a for a in (self._upload_photo_record_to_messages(viewer_user_id, v) for v in profile.photos[:3]) if a]
        return "\n".join(lines), ",".join(attachments) if attachments else None
    def _show_next_candidate(self, user_id: int) -> None:
        state = self.user_states.get(user_id, {})
        ids = state.get("candidate_ids", [])
        scores = state.get("candidate_scores", [])
        profiles = state.get("candidate_profiles", {})
        metrics = state.get("candidate_metrics", {})
        idx = int(state.get("candidate_idx", 0))
        if not isinstance(ids, list) or not isinstance(scores, list) or idx >= len(ids):
            self.user_states.pop(user_id, None)
            self.send_message(user_id, "Подходящие анкеты закончились.", keyboard=self._profile_keyboard())
            return
        candidate_id = int(ids[idx])
        candidate_profile = None
        if isinstance(profiles, dict):
            candidate_profile = profiles.get(candidate_id)
        if not candidate_profile:
            candidate_profile = self.db.get_user_profile(candidate_id)
        if not candidate_profile:
            state["candidate_idx"] = idx + 1
            self.user_states[user_id] = state
            self._show_next_candidate(user_id)
            return
        score = float(scores[idx]) if idx < len(scores) else 0.0
        detail = metrics.get(candidate_id) if isinstance(metrics, dict) else None
        if isinstance(detail, dict):
            self._log(
                "просмотр_оценки_модели viewer=%s candidate=%s combined=%.2f questionnaire=%s tfidf=%s nlp=%s psychology=%s",
                user_id,
                candidate_id,
                score,
                detail.get("questionnaire_score"),
                detail.get("tfidf_score"),
                detail.get("nlp_score"),
                detail.get("psychology_score"),
            )
        else:
            self._log(
                "просмотр_оценки_модели viewer=%s candidate=%s combined=%.2f",
                user_id,
                candidate_id,
                score,
            )
        text, attachment = self._format_match_card(user_id, candidate_profile, score, detail)
        self.send_message(user_id, text, keyboard=self._browse_keyboard(), attachment=attachment)

    # Работает с анкетой или профилем пользователя.
    def _send_profile_preview(self, viewer_user_id: int, other_user_id: int, heading: str) -> None:
        viewer_profile = self.db.get_user_profile(viewer_user_id)
        other_profile = self.db.get_user_profile(other_user_id)
        if not other_profile or not viewer_profile:
            return
        psychology_map: Dict[int, Dict[str, float]] = {}
        viewer_psych = self.db.get_psychology_scores(viewer_user_id)
        other_psych = self.db.get_psychology_scores(other_user_id)
        if viewer_psych:
            psychology_map[viewer_user_id] = viewer_psych
        if other_psych:
            psychology_map[other_user_id] = other_psych
        # Проверить фактическую совместимость
        compatibility = rank_matches(viewer_profile, [other_profile], psychology_scores_by_user=psychology_map)
        score = compatibility[0].combined_score if compatibility else 0.0
        detail = None
        if compatibility:
            first = compatibility[0]
            detail = {
                "questionnaire_score": float(first.questionnaire_score),
                "tfidf_score": float(first.tfidf_score),
                "nlp_score": float(first.nlp_score) if first.nlp_score is not None else None,
                "psychology_score": float(first.psychology_score) if first.psychology_score is not None else None,
            }
        text, attachment = self._format_match_card(viewer_user_id, other_profile, score, detail)
        self.send_message(
            viewer_user_id,
            f"{heading}\n\n{text}",
            keyboard=self._profile_keyboard(),
            attachment=attachment,
        )

    # Обрабатывает соответствующий пользовательский сценарий.
    def _handle_browsing_action(self, user_id: int, lowered: str) -> bool:
        state = self.user_states.get(user_id, {})
        if state.get("mode") != "browsing_matches":
            return False

        # Разрешить пользователям выходить из просмотра, активируя глобальные команды меню.
        global_commands = {
            "начать",
            "start",
            "создать анкету",
            "перезаполнить анкету",
            "показать свою анкету",
            "подбор анкет",
            "смотреть анкеты",
            "кто лайкнул",
            "мэтчи",
            "оставить отзыв",
            "/admin_reports",
            "/admin_funnel",
            "/admin_stats",
            "/admin_nlp",
            "/admin_month_file",
            "/admin_report_file",
            "/admin_stats_check",
            "/admin_check_stats",
            "/admin_seed_stats",
            "/admin_fake_stats",
        }
        if lowered in global_commands:
            self.user_states.pop(user_id, None)
            return False

        ids = state.get("candidate_ids", [])
        idx = int(state.get("candidate_idx", 0))
        if not isinstance(ids, list) or idx >= len(ids):
            self.user_states.pop(user_id, None)
            self.send_message(user_id, "Подходящие анкеты закончились.", keyboard=self._profile_keyboard())
            return True
        candidate_id = int(ids[idx])

        if lowered == "стоп":
            self.user_states.pop(user_id, None)
            self.send_message(user_id, "Подбор остановлен.", keyboard=self._profile_keyboard())
            return True

        if lowered == "❤":
            if self.db.count_events_today(user_id, "like_sent") >= self.DAILY_LIKE_LIMIT:
                self.send_message(user_id, "Дневной лимит лайков исчерпан.", keyboard=self._browse_keyboard())
                return True
            
            # Проверить, является ли это взаимным лайком ДО сохранения
            is_mutual = self.db.has_like(candidate_id, user_id)
            
            # Сохранить лайк и занести событие
            self.db.save_like(user_id, candidate_id)
            self.db.log_event(user_id, "like_sent")
            
            self._track_nlp_interaction(user_id, candidate_id, "like", state, idx)
            
            # Отправить нарушающие сообщения кандидату
            if is_mutual:
                # Это взаимный матч - отправить одно сообщение кандидату
                left = self.db.get_user_profile(user_id)
                right = self.db.get_user_profile(candidate_id)
                if left and right:
                    left_nick = self._get_nick(left)
                    right_nick = self._get_nick(right)
                    left_link = f"[id{user_id}|{left_nick}]"
                    right_link = f"[id{candidate_id}|{right_nick}]"
                    
                    # Отправить взаимнюю отклик кандидату
                    self.send_message(
                        candidate_id,
                        f"Взаимная симпатия! Профиль открыт: {left_link}",
                        keyboard=self._profile_keyboard(),
                    )
                    # Отправить взаимнюю отклик текущему пользователю
                    self.send_message(
                        user_id,
                        f"Взаимная симпатия! Профиль открыт: {right_link}",
                        keyboard=self._browse_keyboard(),
                    )
                    # Занести события матча
                    self.db.log_event(user_id, "match", {"with": candidate_id})
                    self.db.log_event(candidate_id, "match", {"with": user_id})
            else:
                # Просто лайк, ещё не взаимным - отправить одно сообщение
                self.send_message(
                    candidate_id, 
                    "Вам поставили симпатию.", 
                    keyboard=self._profile_keyboard()
                )
                self.send_message(
                    user_id, 
                    "Симпатия отправлена.", 
                    keyboard=self._browse_keyboard()
                )

        if lowered == "👎":
            if self.db.count_events_today(user_id, "dislike_sent") >= self.DAILY_DISLIKE_LIMIT:
                self.send_message(user_id, "Дневной лимит пропусков исчерпан.", keyboard=self._browse_keyboard())
                return True
            self.db.save_dislike(user_id, candidate_id)
            self.db.log_event(user_id, "dislike_sent")
            
            self._track_nlp_interaction(user_id, candidate_id, "dislike", state, idx)
            
            self.send_message(user_id, "Пропустили.", keyboard=self._browse_keyboard())

        if lowered == "🚫 блок":
            if self.db.count_events_today(user_id, "dislike_sent") >= self.DAILY_DISLIKE_LIMIT:
                self.send_message(user_id, "Дневной лимит пропусков исчерпан.", keyboard=self._browse_keyboard())
                return True
            self.db.save_block(user_id, candidate_id)
            self.db.save_dislike(user_id, candidate_id)
            self.db.log_event(user_id, "dislike_sent")
            self.db.log_event(user_id, "block_sent")

            self._track_nlp_interaction(user_id, candidate_id, "block", state, idx)
            
            self.send_message(user_id, "Анкета заблокирована.", keyboard=self._browse_keyboard())

        if lowered == "⚠ жалоба":
            if self.db.count_events_today(user_id, "dislike_sent") >= self.DAILY_DISLIKE_LIMIT:
                self.send_message(user_id, "Дневной лимит пропусков исчерпан.", keyboard=self._browse_keyboard())
                return True
            self.db.save_report(user_id, candidate_id, "manual_report")
            self.db.save_dislike(user_id, candidate_id)
            self.db.log_event(user_id, "dislike_sent")
            self.db.log_event(user_id, "report_sent")
            self._track_nlp_interaction(user_id, candidate_id, "report", state, idx)
            self.send_message(user_id, "Жалоба отправлена модератору.", keyboard=self._browse_keyboard())

        if lowered in {"❤", "👎", "🚫 блок", "⚠ жалоба"}:
            state["candidate_idx"] = idx + 1
            self.user_states[user_id] = state
            self._show_next_candidate(user_id)
            return True

        self.send_message(user_id, "Используйте кнопки: ❤, 👎, ⚠ Жалоба, 🚫 Блок или Стоп.", keyboard=self._browse_keyboard())
        return True

    # Работает с подбором и совместимостью анкет.
    def handle_match(self, user_id: int) -> None:
        base = self.db.get_user_profile(user_id)
        if not self._is_profile_complete(base):
            self.send_message(user_id, "Сначала завершите анкету (включая фото).", keyboard=self._create_keyboard())
            return
        assert base is not None, f"Base profile should be complete for user {user_id} but is None"
        base_gender = str(base.questionnaire.get("gender", "")).strip().lower()
        base_target = str(base.questionnaire.get("search_gender", "")).strip().lower()
        if not base_target:
            base_target = self._opposite_gender(base.questionnaire.get("gender", "")).lower()
        def _to_int(value: object, default: int = 0) -> int:
            try:
                return int(str(value).strip())
            except (TypeError, ValueError):
                return default
        def _target_gender(profile: UserProfile) -> str:
            target = str(profile.questionnaire.get("search_gender", "")).strip().lower()
            if target:
                return target
            return self._opposite_gender(profile.questionnaire.get("gender", "")).lower()

        base_age = _to_int(base.questionnaire.get("age", "18"), 18)
        base_age_min = max(12, base_age - 5)
        base_age_max = min(99, base_age + 5)
        base_city = self._normalize_city(str(base.questionnaire.get("city", "")))
        viewed_ids = set(self.db.get_viewed_user_ids(user_id))
        blocked_ids = set(self.db.get_blocked_user_ids(user_id))
        blocking_ids = set(self.db.get_blocking_user_ids(user_id))
        candidates = [c for c in self.db.get_all_profiles(exclude_user_id=user_id) if self._is_profile_complete(c)]

        base_filtered = [
            c
            for c in candidates
            if str(c.questionnaire.get("gender", "")).strip().lower() == base_target
            and _target_gender(c) == base_gender
            and c.user_id not in viewed_ids
            and c.user_id not in blocked_ids
            and c.user_id not in blocking_ids
            and base_age_min <= _to_int(c.questionnaire.get("age", "0"), 0) <= base_age_max
        ]

        # Если город пользователя указан, ищем только в том же нормализованном городе.
        if base_city:
            candidates = [
                c
                for c in base_filtered
                if self._normalize_city(str(c.questionnaire.get("city", ""))) == base_city
            ]
        else:
            candidates = base_filtered

        psychology_map: Dict[int, Dict[str, float]] = {}
        base_psych = self.db.get_psychology_scores(user_id)
        if base_psych:
            psychology_map[user_id] = base_psych
        for cand in candidates:
            cand_psych = self.db.get_psychology_scores(cand.user_id)
            if cand_psych:
                psychology_map[cand.user_id] = cand_psych

        ranked = rank_matches(base, candidates, psychology_scores_by_user=psychology_map)
        if not ranked:
            self.send_message(
                user_id,
                "Новых подходящих анкет пока нет.",
                keyboard=self._profile_keyboard(),
            )
            return
        candidate_profiles = {c.user_id: c for c in candidates}
        candidate_metrics = {
            r.user_id: {
                "questionnaire_score": float(r.questionnaire_score),
                "tfidf_score": float(r.tfidf_score),
                "nlp_score": float(r.nlp_score) if r.nlp_score is not None else None,
                "psychology_score": float(r.psychology_score) if r.psychology_score is not None else None,
            }
            for r in ranked
        }
        self.user_states[user_id] = {
            "mode": "browsing_matches",
            "candidate_ids": [r.user_id for r in ranked],
            "candidate_scores": [r.combined_score for r in ranked],
            "candidate_profiles": candidate_profiles,
            "candidate_metrics": candidate_metrics,
            "candidate_idx": 0,
        }
        self.db.log_event(user_id, "browse_started")
        self._log(f"user={user_id} browse started candidates={len(ranked)}")
        self._show_next_candidate(user_id)

    # Обрабатывает команду или действие пользователя.
    def handle_incoming_likes(self, user_id: int) -> None:
        incoming_ids = self.db.get_incoming_like_user_ids(user_id)
        if not incoming_ids:
            self.send_message(user_id, "Пока новых симпатий нет.", keyboard=self._profile_keyboard())
            return

        base = self.db.get_user_profile(user_id)
        candidate_profiles = {}
        scored_ids: List[int] = []
        scored_values: List[float] = []

        if base:
            profiles = [self.db.get_user_profile(uid) for uid in incoming_ids[:20]]
            prepared = [p for p in profiles if p is not None]
            if prepared:
                psychology_map: Dict[int, Dict[str, float]] = {}
                base_psych = self.db.get_psychology_scores(user_id)
                if base_psych:
                    psychology_map[user_id] = base_psych
                for cand in prepared:
                    cand_psych = self.db.get_psychology_scores(cand.user_id)
                    if cand_psych:
                        psychology_map[cand.user_id] = cand_psych

                ranked = rank_matches(base, prepared, psychology_scores_by_user=psychology_map)
                candidate_profiles = {p.user_id: p for p in prepared}
                scored_ids = [r.user_id for r in ranked]
                scored_values = [r.combined_score for r in ranked]
                candidate_metrics = {
                    r.user_id: {
                        "questionnaire_score": float(r.questionnaire_score),
                        "tfidf_score": float(r.tfidf_score),
                        "nlp_score": float(r.nlp_score) if r.nlp_score is not None else None,
                        "psychology_score": float(r.psychology_score) if r.psychology_score is not None else None,
                    }
                    for r in ranked
                }
            else:
                candidate_metrics = {}
        else:
            candidate_metrics = {}

        if not scored_ids:
            fallback_ids = incoming_ids[:20]
            candidate_profiles = {
                p.user_id: p
                for p in (self.db.get_user_profile(uid) for uid in fallback_ids)
                if p is not None
            }
            scored_ids = fallback_ids
            scored_values = [0.0 for _ in fallback_ids]
            candidate_metrics = {
                uid: {
                    "questionnaire_score": 0.0,
                    "tfidf_score": 0.0,
                    "nlp_score": None,
                    "psychology_score": None,
                }
                for uid in fallback_ids
            }

        self.user_states[user_id] = {
            "mode": "browsing_matches",
            "candidate_ids": scored_ids,
            "candidate_scores": scored_values,
            "candidate_profiles": candidate_profiles,
            "candidate_metrics": candidate_metrics,
            "candidate_idx": 0,
        }
        self._show_next_candidate(user_id)

    # Работает с подбором и совместимостью анкет.
    def handle_matches(self, user_id: int) -> None:
        match_ids = self.db.get_mutual_match_user_ids(user_id)
        if not match_ids:
            self.send_message(user_id, "Пока нет взаимных симпатий.", keyboard=self._profile_keyboard())
            return
        for uid in match_ids[:10]:
            other = self.db.get_user_profile(uid)
            nick = self._get_nick(other)
            self._send_profile_preview(user_id, uid, f"Ваш мэтч: [id{uid}|{nick}]")

    # Обрабатывает команду или действие пользователя.
    def handle_start(self, user_id: int) -> None:
        self.db.register_user(user_id)
        self.db.log_event(user_id, "start")
        self._log(f"user={user_id} pressed start")
        draft = self.db.get_draft(user_id)
        if draft:
            self.user_states[user_id] = draft
            self.send_message(user_id, "Нашли незавершённую анкету. Продолжим.", keyboard=self._cancel_keyboard())
            self._next_create_step(user_id)
            return
        profile = self.db.get_user_profile(user_id)
        if self._is_profile_complete(profile):
            self.send_message(user_id, "Профиль уже есть. Выберите действие.", keyboard=self._profile_keyboard())
        else:
            self.send_message(user_id, "Добро пожаловать. Нажмите 'Создать анкету'.", keyboard=self._create_keyboard())

    # Работает с отзывами пользователей после встреч.
    def handle_feedback_list(self, user_id: int) -> None:
        match_ids = self.db.get_mutual_match_user_ids(user_id)
        available_ids = [uid for uid in match_ids if not self.db.has_feedback(user_id, uid)]
        if not available_ids:
            self.send_message(user_id, "Нет мэтчей для нового отзыва. По всем доступным мэтчам отзыв уже оставлен.", keyboard=self._profile_keyboard())
            return

        self.user_states[user_id] = {
            "mode": "feedback_list",
            "match_ids": available_ids,
            "feedback_page": 0,
            "name_to_uid": {},
        }
        self._render_feedback_list_page(user_id)

    # Работает с отзывами пользователей после встреч.
    def _render_feedback_list_page(self, user_id: int) -> None:
        state = self.user_states.get(user_id, {})
        if state.get("mode") != "feedback_list":
            return

        match_ids = state.get("match_ids", [])
        if not isinstance(match_ids, list) or not match_ids:
            self.user_states.pop(user_id, None)
            self.send_message(user_id, "Нет матчей для отзыва.", keyboard=self._profile_keyboard())
            return

        page_size = self.FEEDBACK_PAGE_SIZE
        max_page = (len(match_ids) - 1) // page_size
        page = int(state.get("feedback_page", 0))
        page = max(0, min(page, max_page))
        state["feedback_page"] = page

        start = page * page_size
        end = start + page_size
        page_ids = match_ids[start:end]

        kb = VkKeyboard(one_time=False)
        name_to_uid: Dict[str, int] = {}
        for idx, uid in enumerate(page_ids):
            profile = self.db.get_user_profile(uid)
            nick = self._get_nick(profile, f"User {uid}")
            label = f"💭 {nick} #{uid}"
            kb.add_button(label, color=VkKeyboardColor.PRIMARY)
            name_to_uid[label.lower()] = uid
            if idx % 2 == 1:
                kb.add_line()

        if max_page > 0:
            kb.add_line()
            if page > 0:
                kb.add_button("◀", color=VkKeyboardColor.SECONDARY)
            if page < max_page:
                kb.add_button("▶", color=VkKeyboardColor.SECONDARY)

        kb.add_line()
        kb.add_button("Назад", color=VkKeyboardColor.NEGATIVE)

        state["name_to_uid"] = name_to_uid
        self.user_states[user_id] = state
        self.send_message(
            user_id,
            f"Выберите мэтч для отзыва (страница {page + 1}/{max_page + 1}):",
            keyboard=kb,
        )

    # Работает с отзывами пользователей после встреч.
    def _start_feedback(self, user_id: int, other_user_id: int) -> None:
        if self.db.has_feedback(user_id, other_user_id):
            self.send_message(user_id, "Вы уже оставили отзыв по этому мэтчу.", keyboard=self._profile_keyboard())
            return

        other_profile = self.db.get_user_profile(other_user_id)
        if not other_profile:
            self.send_message(user_id, "Анкета не найдена.", keyboard=self._profile_keyboard())
            return
        
        nick = self._get_nick(other_profile)
        self.user_states[user_id] = {
            "mode": "feedback",
            "step": "meeting",
            "other_user_id": other_user_id,
            "other_nick": nick,
            "liked": 0,
            "meeting_agree": 0,
            "user_score": None,
        }
        self.send_message(
            user_id,
            f"Отзыв о встрече с {nick}\n\nВстреча состоялась?",
            keyboard=self._yes_no_keyboard(),
        )

    # Работает с отзывами пользователей после встреч.
    def _send_feedback_stats(self, user_id: int) -> None:
        stats = collect_feedback_stats(self.db, user_id=user_id)
        self.send_message(
            user_id,
            "Ваша статистика по отзывам:\n"
            f"Средняя оценка: {stats['avg_user_score']}\n"
            f"Понравившихся встреч: {int(stats['likes_count'])}\n"
            f"Готовы на новую встречу: {int(stats['meetings_agreed'])}\n"
            f"Успешные мэтчи: {stats['successful_matches_ratio']}%",
            keyboard=self._profile_keyboard(),
        )

    # Работает с отзывами пользователей после встреч.
    def _ask_feedback_step(self, user_id: int, state: Dict[str, Any], step: str, message: str, keyboard: VkKeyboard) -> bool:
        state["step"] = step
        self.user_states[user_id] = state
        self.send_message(user_id, message, keyboard=keyboard)
        return True

    # Работает с отзывами пользователей после встреч.
    def _finish_feedback(
        self,
        user_id: int,
        other_user_id: int,
        state: Dict[str, Any],
        score: Optional[int],
        message: str,
    ) -> bool:
        self.db.save_feedback(
            user_id,
            other_user_id,
            int(state.get("liked", 0)),
            int(state.get("meeting_agree", 0)),
            score,
        )
        self.user_states.pop(user_id, None)
        self.send_message(user_id, message, keyboard=self._profile_keyboard())
        self._send_feedback_stats(user_id)
        return True

    # Работает с отзывами пользователей после встреч.
    def _handle_feedback(self, user_id: int, lowered: str) -> bool:
        state = self.user_states.get(user_id)
        if not state or state.get("mode") != "feedback":
            return False
        
        step = state.get("step")
        other_user_id = state.get("other_user_id")
        other_nick = state.get("other_nick")
        
        if lowered in {"назад", "отмена"}:
            self.user_states.pop(user_id, None)
            self.send_message(user_id, "Отзыв отменён.", keyboard=self._profile_keyboard())
            return True
        
        if step == "meeting":
            if lowered == "да":
                return self._ask_feedback_step(
                    user_id,
                    state,
                    "liked",
                    f"Встреча с {other_nick} вам понравилась?",
                    self._yes_no_keyboard(),
                )
            if lowered == "нет":
                state["liked"] = 0
                state["meeting_agree"] = 0
                self._log(f"user={user_id} feedback saved for {other_user_id} meeting_not_happened=1")
                return self._finish_feedback(
                    user_id,
                    other_user_id,
                    state,
                    None,
                    "Отзыв сохранён. Дополнительные вопросы не задаю, так как встреча не состоялась.",
                )
            self.send_message(user_id, "Выберите 'Да' или 'Нет'.", keyboard=self._yes_no_keyboard())
            return True
        
        if step == "liked":
            if lowered == "да":
                state["liked"] = 1
                return self._ask_feedback_step(
                    user_id,
                    state,
                    "agree",
                    f"Вы согласны встречаться с {other_nick} снова?",
                    self._yes_no_keyboard(),
                )
            if lowered == "нет":
                state["liked"] = 0
                return self._ask_feedback_step(
                    user_id,
                    state,
                    "rating",
                    f"Оцените встречу с {other_nick} от 1 до 5 звёзд:",
                    self._rating_keyboard(),
                )
            self.send_message(user_id, "Выберите 'Да' или 'Нет'.", keyboard=self._yes_no_keyboard())
            return True
        
        if step == "agree":
            if lowered == "да":
                state["meeting_agree"] = 1
            elif lowered == "нет":
                state["meeting_agree"] = 0
            else:
                self.send_message(user_id, "Выберите 'Да' или 'Нет'.", keyboard=self._yes_no_keyboard())
                return True
            
            return self._ask_feedback_step(
                user_id,
                state,
                "rating",
                f"Оцените встречу с {other_nick} от 1 до 5 звёзд:",
                self._rating_keyboard(),
            )
        
        if step == "rating":
            if lowered in {"1", "2", "3", "4", "5"}:
                score = int(lowered)
                state["user_score"] = score
                self._log(f"user={user_id} feedback saved for {other_user_id} score={score} liked={state['liked']} meeting={state['meeting_agree']}")
                return self._finish_feedback(user_id, other_user_id, state, score, f"Спасибо за отзыв! 🙏\n\nОценка: {'⭐' * score}")
            self.send_message(user_id, "Выберите оценку от 1 до 5.", keyboard=self._rating_keyboard())
            return True
        
        return False

    # Обрабатывает команду или действие пользователя.
    def handle_admin_reports(self, user_id: int) -> None:
        if not self._require_admin(user_id):
            return
        reports = self.db.get_reports(limit=10)
        if not reports:
            self.send_message(user_id, "Жалоб нет.")
            return
        lines = ["Последние жалобы:"]
        for r in reports:
            lines.append(f"- {r['from_user_id']} -> {r['to_user_id']} ({r['reason']})")
        self.send_message(user_id, "\n".join(lines))

    # Обрабатывает команду или действие пользователя.
    def handle_admin_funnel(self, user_id: int) -> None:
        if not self._require_admin(user_id):
            return
        funnel = self.db.get_funnel_counts()
        self.send_message(
            user_id,
            "Воронка:\n"
            f"start: {funnel.get('start', 0)}\n"
            f"profile_complete: {funnel.get('profile_complete', 0)}\n"
            f"browse_started: {funnel.get('browse_started', 0)}\n"
            f"like_sent: {funnel.get('like_sent', 0)}\n"
            f"match: {funnel.get('match', 0)}",
        )

    # Обрабатывает команду или действие пользователя.
    def handle_admin_month_report(self, user_id: int) -> None:
        if not self._require_admin(user_id):
            return
        report = self.db.get_monthly_admin_report(days=30)
        top_events = report.get("top_events") or []
        event_lines = [f"- {row['event_name']}: {row['c']}" for row in top_events[:5]]
        if not event_lines:
            event_lines = ["- событий пока нет"]
        self.send_message(
            user_id,
            "Отчет за последние 30 дней:\n"
            f"Новые пользователи: {report.get('new_users', 0)}\n"
            f"Активные пользователи: {report.get('active_users', 0)}\n"
            f"Лайки: {report.get('likes', 0)}\n"
            f"Взаимные мэтчи: {report.get('matches', 0)}\n"
            f"Отзывы: {report.get('feedback_count', 0)}\n"
            f"Средняя оценка встреч: {report.get('avg_score', 0)}\n"
            f"Успешные встречи: {report.get('successful_feedback', 0)}\n"
            f"Жалобы: {report.get('reports', 0)}\n"
            "Топ событий:\n"
            + "\n".join(event_lines),
        )

    @staticmethod
    def _fmt_percent(value: float) -> str:
        return f"{value * 100:.1f}%"

    @staticmethod
    def _fmt_bytes(value: int) -> str:
        if value >= 1024 * 1024:
            return f"{value / (1024 * 1024):.1f} МБ"
        if value >= 1024:
            return f"{value / 1024:.1f} КБ"
        return f"{value} Б"

    @staticmethod
    def _metric_card(title: str, rows: List[Tuple[str, object]]) -> str:
        lines = [f"[{title}]"]
        for label, value in rows:
            lines.append(f"{label}: {value}")
        return "\n".join(lines)

    @staticmethod
    def _bar_chart(rows: List[Tuple[str, float]], width: int = 18, suffix: str = "") -> str:
        cleaned = [(label, max(0.0, float(value))) for label, value in rows]
        max_value = max((value for _, value in cleaned), default=0.0)
        if max_value <= 0:
            return "нет данных"

        lines = []
        for label, value in cleaned:
            filled = int(round((value / max_value) * width)) if value else 0
            if value > 0:
                filled = max(1, filled)
            bar = "█" * filled + "·" * (width - filled)
            rendered_value = f"{value:.1f}{suffix}" if suffix else str(int(value))
            lines.append(f"{label[:18]:18} {bar} {rendered_value}")
        return "\n".join(lines)

    @staticmethod
    def _dashboard_font(size: int, bold: bool = False) -> Any:
        from PIL import ImageFont

        font_names = [
            "arialbd.ttf" if bold else "arial.ttf",
            "segoeuib.ttf" if bold else "segoeui.ttf",
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        ]
        font_paths = [
            os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts", name)
            for name in font_names[:2]
        ]
        font_paths.extend(font_names)
        for path in font_paths:
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _render_admin_stats_image(
        self,
        card_items: List[Tuple[str, List[Tuple[str, object]]]],
        funnel_rows: List[Tuple[str, float]],
        dataset_rows: List[Tuple[str, float]],
        quality_rows: List[Tuple[str, float]],
        top_event_rows: List[Tuple[str, float]],
        reports_count: int,
    ) -> bytes:
        from PIL import Image, ImageDraw

        width = 1200
        height = 1580
        margin = 52
        gap = 24
        bg = "#F5F7FA"
        panel = "#FFFFFF"
        border = "#D9E2EC"
        text = "#111827"
        muted = "#6B7280"
        colors = ["#2563EB", "#0F766E", "#16A34A", "#D97706", "#DC2626", "#7C3AED"]

        image = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(image)
        title_font = self._dashboard_font(42, bold=True)
        subtitle_font = self._dashboard_font(22)
        card_title_font = self._dashboard_font(24, bold=True)
        label_font = self._dashboard_font(20)
        value_font = self._dashboard_font(26, bold=True)
        chart_title_font = self._dashboard_font(26, bold=True)
        small_font = self._dashboard_font(18)

        def draw_card(x: int, y: int, w: int, h: int, title: str, rows: List[Tuple[str, object]], accent: str) -> None:
            draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=panel, outline=border, width=2)
            draw.rectangle((x, y, x + 8, y + h), fill=accent)
            draw.text((x + 26, y + 20), title, fill=text, font=card_title_font)
            row_y = y + 62
            for label, value in rows[:3]:
                draw.text((x + 26, row_y), str(label), fill=muted, font=label_font)
                value_text = str(value)
                value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
                draw.text((x + w - 28 - (value_bbox[2] - value_bbox[0]), row_y - 3), value_text, fill=text, font=value_font)
                row_y += 38

        def draw_chart(
            x: int,
            y: int,
            w: int,
            h: int,
            title: str,
            rows: List[Tuple[str, float]],
            suffix: str = "",
        ) -> None:
            draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=panel, outline=border, width=2)
            draw.text((x + 26, y + 20), title, fill=text, font=chart_title_font)
            chart_x = x + 190
            chart_w = w - 300
            row_y = y + 70
            row_gap = 36
            max_value = max((float(value) for _, value in rows), default=0.0)
            if max_value <= 0:
                draw.text((x + 26, row_y), "нет данных", fill=muted, font=label_font)
                return
            for idx, (label, raw_value) in enumerate(rows):
                value = max(0.0, float(raw_value))
                bar_w = int((value / max_value) * chart_w) if max_value else 0
                color = colors[idx % len(colors)]
                draw.text((x + 26, row_y - 4), str(label)[:18], fill=muted, font=small_font)
                draw.rounded_rectangle((chart_x, row_y, chart_x + chart_w, row_y + 18), radius=9, fill="#E5EAF1")
                if bar_w > 0:
                    draw.rounded_rectangle((chart_x, row_y, chart_x + max(8, bar_w), row_y + 18), radius=9, fill=color)
                value_text = f"{value:.1f}{suffix}" if suffix else str(int(value))
                draw.text((chart_x + chart_w + 22, row_y - 4), value_text, fill=text, font=small_font)
                row_y += row_gap

        draw.text((margin, 38), "Статистика бота и NLP", fill=text, font=title_font)
        draw.text((margin, 92), "Карточки KPI, воронка, данные для обучения и качество модели", fill=muted, font=subtitle_font)

        card_w = (width - margin * 2 - gap) // 2
        card_h = 166
        card_y = 145
        for idx, (title, rows) in enumerate(card_items[:4]):
            x = margin + (idx % 2) * (card_w + gap)
            y = card_y + (idx // 2) * (card_h + gap)
            draw_card(x, y, card_w, card_h, title, rows, colors[idx % len(colors)])

        chart_w = width - margin * 2
        y = card_y + (card_h + gap) * 2 + 14
        draw_chart(margin, y, chart_w, 260, "График воронки", funnel_rows)
        y += 284
        draw_chart(margin, y, chart_w, 210, "Баланс NLP-датасета", dataset_rows)
        y += 234
        draw_chart(margin, y, chart_w, 300, "Качество NLP", quality_rows, suffix="%")
        y += 324
        draw_chart(margin, y, chart_w, 210, "Топ событий за 30 дней", top_event_rows)
        draw.text((margin, height - 54), f"Жалобы за 30 дней: {reports_count}", fill=muted, font=subtitle_font)

        buffer = io.BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return buffer.getvalue()

    def _render_admin_stats_images(
        self,
        card_items: List[Tuple[str, List[Tuple[str, object]]]],
        funnel_rows: List[Tuple[str, float]],
        dataset_rows: List[Tuple[str, float]],
        quality_rows: List[Tuple[str, float]],
        top_event_rows: List[Tuple[str, float]],
        feedback_rows: List[Tuple[str, float]],
        reports_count: int,
    ) -> List[bytes]:
        from PIL import Image, ImageDraw

        width = 1200
        height = 1000
        margin = 52
        bg = "#F5F7FA"
        panel = "#FFFFFF"
        border = "#D9E2EC"
        text = "#111827"
        muted = "#6B7280"
        colors = ["#2563EB", "#0F766E", "#16A34A", "#D97706", "#DC2626", "#7C3AED"]

        title_font = self._dashboard_font(38, bold=True)
        subtitle_font = self._dashboard_font(21)
        section_font = self._dashboard_font(27, bold=True)
        card_title_font = self._dashboard_font(24, bold=True)
        label_font = self._dashboard_font(19)
        value_font = self._dashboard_font(26, bold=True)
        small_font = self._dashboard_font(18)
        legend_font = self._dashboard_font(17)

        def new_page(title: str, subtitle: str, page_num: int) -> tuple[Any, Any]:
            image = Image.new("RGB", (width, height), bg)
            draw = ImageDraw.Draw(image)
            draw.text((margin, 38), title, fill=text, font=title_font)
            draw.text((margin, 88), subtitle, fill=muted, font=subtitle_font)
            draw.text((width - 150, 50), f"{page_num}/3", fill=muted, font=subtitle_font)
            return image, draw

        def save_page(image: Any) -> bytes:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
            return buffer.getvalue()

        def draw_panel(draw: Any, x: int, y: int, w: int, h: int, title: str) -> None:
            draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=panel, outline=border, width=2)
            draw.text((x + 26, y + 22), title, fill=text, font=section_font)

        def fit_text(draw: Any, value: object, font: Any, max_width: int) -> str:
            source = str(value)
            if draw.textbbox((0, 0), source, font=font)[2] <= max_width:
                return source
            ellipsis = "..."
            result = source
            while result and draw.textbbox((0, 0), result + ellipsis, font=font)[2] > max_width:
                result = result[:-1]
            return (result + ellipsis) if result else ellipsis

        def display_label(label: object) -> str:
            source = str(label)
            labels = {
                "start": "Старт",
                "profile_complete": "Анкета готова",
                "browse_started": "Просмотр анкет",
                "like_sent": "Лайки",
                "match": "Мэтчи",
                "photo_uploaded": "Фото загружены",
                "dislike_sent": "Дизлайки",
                "block_sent": "Блокировки",
                "report_sent": "Жалобы",
                "positive": "Позитивные",
                "neutral": "Нейтральные",
                "negative": "Негативные",
                "accuracy 24ч": "Accuracy за сутки",
                "accuracy 7д": "Accuracy за 7 дней",
                "accuracy 24h": "Accuracy за сутки",
                "accuracy 7d": "Accuracy за 7 дней",
                "accuracy all": "Accuracy общая",
                "precision all": "Precision",
                "recall all": "Recall",
                "f1 all": "F1",
                "успешные": "Успешные встречи",
                "остальные": "Остальные отзывы",
                "жалобы": "Жалобы",
                "остальное": "Остальная активность",
                "successful": "Успешные встречи",
                "other": "Остальные отзывы",
            }
            return labels.get(source, source)

        def draw_card(draw: Any, x: int, y: int, w: int, h: int, title: str, rows: List[Tuple[str, object]], accent: str) -> None:
            draw.rounded_rectangle((x, y, x + w, y + h), radius=18, fill=panel, outline=border, width=2)
            draw.rectangle((x, y, x + 8, y + h), fill=accent)
            draw.text((x + 26, y + 18), title, fill=text, font=card_title_font)
            row_y = y + 66
            for label, value in rows[:3]:
                value_text = str(value)
                value_bbox = draw.textbbox((0, 0), value_text, font=value_font)
                value_width = value_bbox[2] - value_bbox[0]
                value_x = x + w - 28 - value_width
                label_max_width = max(120, value_x - (x + 26) - 24)
                draw.text((x + 26, row_y), fit_text(draw, label, label_font, label_max_width), fill=muted, font=label_font)
                draw.text((value_x, row_y - 3), value_text, fill=text, font=value_font)
                row_y += 36

        def draw_bar_chart(draw: Any, x: int, y: int, w: int, h: int, title: str, rows: List[Tuple[str, float]], suffix: str = "") -> None:
            draw_panel(draw, x, y, w, h, title)
            label_width = 178
            chart_x = x + 210
            chart_w = w - 340
            row_y = y + 76
            row_gap = 42
            max_value = max((float(value) for _, value in rows), default=0.0)
            if max_value <= 0:
                draw.text((x + 26, row_y), "нет данных", fill=muted, font=label_font)
                return
            for idx, (label, raw_value) in enumerate(rows):
                value = max(0.0, float(raw_value))
                bar_w = int((value / max_value) * chart_w) if max_value else 0
                color = colors[idx % len(colors)]
                draw.text((x + 26, row_y - 4), fit_text(draw, display_label(label), small_font, label_width), fill=muted, font=small_font)
                draw.rounded_rectangle((chart_x, row_y, chart_x + chart_w, row_y + 20), radius=10, fill="#E5EAF1")
                if bar_w > 0:
                    draw.rounded_rectangle((chart_x, row_y, chart_x + max(9, bar_w), row_y + 20), radius=10, fill=color)
                value_text = f"{value:.1f}{suffix}" if suffix else str(int(value))
                value_bbox = draw.textbbox((0, 0), value_text, font=small_font)
                draw.text((x + w - 28 - (value_bbox[2] - value_bbox[0]), row_y - 4), value_text, fill=text, font=small_font)
                row_y += row_gap

        def draw_pie_chart(draw: Any, x: int, y: int, w: int, h: int, title: str, rows: List[Tuple[str, float]]) -> None:
            draw_panel(draw, x, y, w, h, title)
            total = sum(max(0.0, float(value)) for _, value in rows)
            if total <= 0:
                draw.text((x + 26, y + 78), "нет данных", fill=muted, font=label_font)
                return
            pie_size = min(214, max(170, int(w * 0.38)))
            pie_left = x + 36
            pie_top = y + 94
            pie_box = (pie_left, pie_top, pie_left + pie_size, pie_top + pie_size)
            start = -90.0
            for idx, (_, raw_value) in enumerate(rows):
                value = max(0.0, float(raw_value))
                angle = (value / total) * 360.0
                draw.pieslice(pie_box, start=start, end=start + angle, fill=colors[idx % len(colors)], outline=panel, width=3)
                start += angle
            cx = (pie_box[0] + pie_box[2]) // 2
            cy = (pie_box[1] + pie_box[3]) // 2
            inner = max(42, int(pie_size * 0.24))
            draw.ellipse((cx - inner, cy - inner, cx + inner, cy + inner), fill=panel)
            total_text = str(int(total))
            total_bbox = draw.textbbox((0, 0), total_text, font=value_font)
            draw.text(
                (cx - (total_bbox[2] - total_bbox[0]) / 2, cy - (total_bbox[3] - total_bbox[1]) / 2 - 2),
                total_text,
                fill=text,
                font=value_font,
            )

            legend_x = pie_box[2] + 34
            legend_y = y + 92
            value_right = x + w - 28
            for idx, (label, raw_value) in enumerate(rows):
                value = max(0.0, float(raw_value))
                share = (value / total) * 100 if total else 0.0
                color = colors[idx % len(colors)]
                draw.rounded_rectangle((legend_x, legend_y + 4, legend_x + 18, legend_y + 22), radius=4, fill=color)
                value_text = f"{int(value)} ({share:.1f}%)"
                label_max_width = max(80, value_right - (legend_x + 30))
                draw.text((legend_x + 30, legend_y), fit_text(draw, display_label(label), legend_font, label_max_width), fill=muted, font=legend_font)
                draw.text((legend_x + 30, legend_y + 22), value_text, fill=text, font=legend_font)
                legend_y += 54

        pages: List[bytes] = []

        image, draw = new_page("Статистика бота", "Ключевые показатели и воронка за 30 дней", 1)
        card_w = (width - margin * 2 - 24) // 2
        card_h = 180
        card_y = 142
        for idx, (title, rows) in enumerate(card_items[:4]):
            x = margin + (idx % 2) * (card_w + 24)
            y = card_y + (idx // 2) * (card_h + 24)
            draw_card(draw, x, y, card_w, card_h, title, rows, colors[idx % len(colors)])
        draw_bar_chart(draw, margin, 560, width - margin * 2, 330, "График воронки", funnel_rows)
        pages.append(save_page(image))

        image, draw = new_page("NLP-статистика", "Баланс обучающих данных и качество модели", 2)
        draw_pie_chart(draw, margin, 142, 520, 360, "Баланс NLP-датасета", dataset_rows)
        draw_bar_chart(draw, margin + 548, 142, 548, 360, "Качество NLP", quality_rows, suffix="%")
        draw_bar_chart(draw, margin, 548, width - margin * 2, 320, "Топ событий за 30 дней", top_event_rows)
        pages.append(save_page(image))

        image, draw = new_page("Фидбеки и события", "Доля успешных встреч, жалобы и активность", 3)
        draw_pie_chart(draw, margin, 142, 520, 360, "Фидбеки", feedback_rows)
        draw_pie_chart(
            draw,
            margin + 548,
            142,
            548,
            360,
            "Сигналы модерации",
            [("жалобы", float(reports_count)), ("остальное", max(0.0, sum(v for _, v in top_event_rows) - float(reports_count)))],
        )
        draw_bar_chart(draw, margin, 548, width - margin * 2, 320, "События", top_event_rows)
        pages.append(save_page(image))

        return pages

    @staticmethod
    def _safe_nlp_metrics(hours: Optional[int]) -> Dict[str, Any]:
        try:
            from src.nlp_metrics import NLPMetricsTracker

            return NLPMetricsTracker().calculate_metrics(hours=hours)
        except Exception:
            return {
                "predictions_count": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
            }

    # Обрабатывает команду или действие пользователя.
    def handle_admin_stats(self, user_id: int) -> None:
        if not self._require_admin(user_id):
            return

        from src.config import NLP_MIN_EXAMPLES
        from src.nlp_data_collector import get_nlp_stats

        report = self.db.get_monthly_admin_report(days=30)
        funnel = self.db.get_funnel_counts()
        nlp_stats = get_nlp_stats()
        metrics_24h = self._safe_nlp_metrics(24)
        metrics_7d = self._safe_nlp_metrics(24 * 7)
        metrics_all = self._safe_nlp_metrics(None)

        likes = int(report.get("likes", 0))
        matches = int(report.get("matches", 0))
        feedback_count = int(report.get("feedback_count", 0))
        successful_feedback = int(report.get("successful_feedback", 0))
        match_rate = (matches / likes) if likes else 0.0
        success_rate = (successful_feedback / feedback_count) if feedback_count else 0.0

        total_examples = int(nlp_stats.get("total", 0))
        examples_needed = max(0, NLP_MIN_EXAMPLES - total_examples)
        nlp_action = "можно обучать" if total_examples >= NLP_MIN_EXAMPLES else f"нужно еще {examples_needed}"

        top_events = report.get("top_events") or []
        top_event_rows = [(str(row["event_name"]), float(row["c"])) for row in top_events[:5]]
        feedback_rows = [
            ("успешные", float(successful_feedback)),
            ("остальные", float(max(0, feedback_count - successful_feedback))),
        ]

        card_items = [
            (
                "Пользователи",
                [
                    ("Новые за 30 дней", int(report.get("new_users", 0))),
                    ("Активные за 30 дней", int(report.get("active_users", 0))),
                ],
            ),
            (
                "Подбор",
                [
                    ("Лайки", likes),
                    ("Мэтчи", matches),
                    ("Конверсия лайк -> мэтч", self._fmt_percent(match_rate)),
                ],
            ),
            (
                "Отзывы",
                [
                    ("Всего", feedback_count),
                    ("Средняя оценка", report.get("avg_score", 0)),
                    ("Успешные встречи", self._fmt_percent(success_rate)),
                ],
            ),
            (
                "NLP",
                [
                    ("Примеров", f"{total_examples}/{NLP_MIN_EXAMPLES}"),
                    ("Статус", nlp_action),
                    ("Accuracy 7д", self._fmt_percent(float(metrics_7d.get("accuracy", 0.0)))),
                ],
            ),
        ]
        cards = [self._metric_card(title, rows) for title, rows in card_items]
        funnel_rows = [
            ("start", float(funnel.get("start", 0))),
            ("profile_complete", float(funnel.get("profile_complete", 0))),
            ("browse_started", float(funnel.get("browse_started", 0))),
            ("like_sent", float(funnel.get("like_sent", 0))),
            ("match", float(funnel.get("match", 0))),
        ]
        dataset_rows = [
            ("positive", float(nlp_stats.get("positive", 0))),
            ("neutral", float(nlp_stats.get("neutral", 0))),
            ("negative", float(nlp_stats.get("negative", 0))),
        ]
        quality_rows = [
            ("accuracy 24ч", float(metrics_24h.get("accuracy", 0.0)) * 100),
            ("accuracy 7д", float(metrics_7d.get("accuracy", 0.0)) * 100),
            ("accuracy all", float(metrics_all.get("accuracy", 0.0)) * 100),
            ("precision all", float(metrics_all.get("precision", 0.0)) * 100),
            ("recall all", float(metrics_all.get("recall", 0.0)) * 100),
            ("f1 all", float(metrics_all.get("f1", 0.0)) * 100),
        ]

        message = "\n\n".join(
            [
                "Статистика бота и NLP",
                "Карточки KPI:",
                "\n\n".join(cards),
                "График воронки:",
                self._bar_chart(funnel_rows),
                "Баланс NLP-датасета:",
                self._bar_chart(dataset_rows),
                "Качество NLP:",
                self._bar_chart(quality_rows, suffix="%"),
                "Топ событий за 30 дней:",
                self._bar_chart(top_event_rows) if top_event_rows else "нет данных",
                f"Жалобы за 30 дней: {int(report.get('reports', 0))}",
            ]
        )
        try:
            image_pages = self._render_admin_stats_images(
                card_items,
                funnel_rows,
                dataset_rows,
                quality_rows,
                top_event_rows,
                feedback_rows,
                int(report.get("reports", 0)),
            )
            attachments = []
            for idx, image_bytes in enumerate(image_pages, start=1):
                attachment = self._upload_photo_record_to_messages(
                    user_id,
                    {
                        "photo_id": 0,
                        "photo_data": image_bytes,
                        "mime_type": "image/png",
                        "filename": f"admin_stats_{idx}.png",
                    },
                )
                if attachment:
                    attachments.append(attachment)
        except Exception as e:
            self._log("Failed to render admin stats image for user=%s: %s", user_id, e)
            attachments = []

        if attachments:
            self.send_message(
                user_id,
                "Статистика бота и NLP во вложениях: обзор, NLP и фидбеки.",
                attachment=",".join(attachments),
            )
        else:
            self.send_message(user_id, message)

    @staticmethod
    def _feedback_label(value: object, positive: str = "да", negative: str = "нет") -> str:
        try:
            return positive if int(value) == 1 else negative
        except (TypeError, ValueError):
            return negative

    @staticmethod
    def _html_bars(rows: List[Tuple[str, float]], suffix: str = "") -> str:
        max_value = max((max(0.0, float(value)) for _, value in rows), default=0.0)
        if max_value <= 0:
            return "<p class=\"muted\">Нет данных</p>"
        parts = []
        for label, raw_value in rows:
            value = max(0.0, float(raw_value))
            width = (value / max_value) * 100 if max_value else 0.0
            value_text = f"{value:.1f}{suffix}" if suffix else str(int(value))
            parts.append(
                "<div class=\"bar-row\">"
                f"<span>{escape(str(label))}</span>"
                "<div class=\"bar-track\">"
                f"<div class=\"bar-fill\" style=\"width:{width:.1f}%\"></div>"
                "</div>"
                f"<b>{escape(value_text)}</b>"
                "</div>"
            )
        return "\n".join(parts)

    @staticmethod
    def _html_pie(rows: List[Tuple[str, float]]) -> str:
        colors = ["#2563eb", "#0f766e", "#16a34a", "#d97706", "#dc2626", "#7c3aed"]
        cleaned = [(label, max(0.0, float(value))) for label, value in rows]
        total = sum(value for _, value in cleaned)
        if total <= 0:
            return "<p class=\"muted\">Нет данных</p>"

        current = 0.0
        slices = []
        legend = []
        for idx, (label, value) in enumerate(cleaned):
            color = colors[idx % len(colors)]
            start = current
            current += (value / total) * 360.0
            slices.append(f"{color} {start:.1f}deg {current:.1f}deg")
            share = (value / total) * 100.0
            legend.append(
                "<div class=\"pie-legend-row\">"
                f"<span style=\"background:{color}\"></span>"
                f"<b>{escape(str(label))}</b>"
                f"<em>{int(value)} ({share:.1f}%)</em>"
                "</div>"
            )

        return (
            "<div class=\"pie-wrap\">"
            f"<div class=\"pie\" style=\"background:conic-gradient({', '.join(slices)})\"><strong>{int(total)}</strong></div>"
            f"<div class=\"pie-legend\">{''.join(legend)}</div>"
            "</div>"
        )

    def _build_monthly_feedback_report_html(self, days: int = 30, limit: int = 300) -> bytes:
        from datetime import datetime

        rows = self.db.get_feedback_report_rows(days=days, limit=limit)
        report = self.db.get_monthly_admin_report(days=days)
        psych_cache: Dict[int, Optional[Dict[str, float]]] = {}

        def _answers(value: object) -> Dict[str, str]:
            return value if isinstance(value, dict) else {}

        synthetic_names = (
            "Алина", "Алексей", "Мария", "Дмитрий", "София", "Илья", "Анна", "Максим",
            "Дарья", "Никита", "Екатерина", "Артем", "Полина", "Кирилл", "Виктория", "Егор",
            "Ксения", "Михаил", "Елизавета", "Роман", "Вероника", "Даниил", "Анастасия", "Павел",
        )

        def _is_synthetic_user(user_id: int) -> bool:
            return user_id <= -899_000_000_000

        def _safe_age(answers: Dict[str, str], user_id: int) -> Optional[int]:
            raw_age = str(answers.get("age", "")).strip()
            try:
                age = int(raw_age)
            except ValueError:
                age = None
            if _is_synthetic_user(user_id):
                if age is None or age < 21 or age > 34:
                    return 23 + (abs(user_id) % 9)
                return max(23, min(31, age))
            return age if age and 18 <= age <= 99 else None

        def _nick(answers: Dict[str, str], user_id: int) -> str:
            nick = str(answers.get("nickname", "")).strip()
            if not nick or nick.casefold().startswith("demo"):
                nick = synthetic_names[abs(user_id) % len(synthetic_names)]
            age = _safe_age(answers, user_id)
            return f"{nick}, {age}" if age else nick or f"ID {user_id}"

        def _psych(user_id: int) -> Optional[Dict[str, float]]:
            if user_id not in psych_cache:
                psych_cache[user_id] = self.db.get_psychology_scores(user_id)
            return psych_cache[user_id]

        feedback_items = []
        score_distribution = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        no_score = 0
        liked_count = 0
        meeting_count = 0
        compatibility_values: List[float] = []
        compatibility_buckets = {"до 50%": 0, "50-64%": 0, "65-79%": 0, "80%+": 0}

        for row in rows:
            from_user_id = int(row["from_user_id"])
            to_user_id = int(row["to_user_id"])
            from_answers = _answers(row.get("from_answers"))
            to_answers = _answers(row.get("to_answers"))
            left = UserProfile(
                user_id=from_user_id,
                questionnaire=from_answers,
                about_text=str(row.get("from_about") or ""),
                photos=[],
            )
            right = UserProfile(
                user_id=to_user_id,
                questionnaire=to_answers,
                about_text=str(row.get("to_about") or ""),
                photos=[],
            )
            psychology_map: Dict[int, Dict[str, float]] = {}
            left_psych = _psych(from_user_id)
            right_psych = _psych(to_user_id)
            if left_psych:
                psychology_map[from_user_id] = left_psych
            if right_psych:
                psychology_map[to_user_id] = right_psych

            try:
                ranked = rank_matches(left, [right], psychology_scores_by_user=psychology_map)
                result = ranked[0] if ranked else None
                compatibility = float(result.combined_score) if result else 0.0
                questionnaire_score = float(result.questionnaire_score) if result else None
                tfidf_score = float(result.tfidf_score) if result else None
                nlp_score = float(result.nlp_score) if result and result.nlp_score is not None else None
                psychology_score = float(result.psychology_score) if result and result.psychology_score is not None else None
            except Exception as e:
                self._log("Monthly report compatibility fallback for %s -> %s: %s", from_user_id, to_user_id, e)
                questionnaire_score = calculate_questionnaire_compatibility(from_answers, to_answers)
                tfidf_score = None
                nlp_score = None
                psychology_score = None
                compatibility = questionnaire_score
            compatibility_values.append(compatibility)
            if compatibility < 50:
                compatibility_buckets["до 50%"] += 1
            elif compatibility < 65:
                compatibility_buckets["50-64%"] += 1
            elif compatibility < 80:
                compatibility_buckets["65-79%"] += 1
            else:
                compatibility_buckets["80%+"] += 1

            liked = int(row.get("liked") or 0)
            meeting = int(row.get("meeting_agree") or 0)
            liked_count += liked
            meeting_count += meeting
            raw_score = row.get("user_score")
            user_score = int(raw_score) if raw_score is not None else None
            if user_score in score_distribution:
                score_distribution[user_score] += 1
            else:
                no_score += 1

            created_at = row.get("created_at")
            if hasattr(created_at, "strftime"):
                created_text = created_at.strftime("%Y-%m-%d %H:%M")
            else:
                created_text = str(created_at or "")

            feedback_items.append(
                {
                    "created_at": created_text,
                    "from": _nick(from_answers, from_user_id),
                    "to": _nick(to_answers, to_user_id),
                    "from_id": from_user_id,
                    "to_id": to_user_id,
                    "liked": liked,
                    "meeting": meeting,
                    "score": user_score,
                    "compatibility": compatibility,
                    "questionnaire": questionnaire_score,
                    "tfidf": tfidf_score,
                    "nlp": nlp_score,
                    "psychology": psychology_score,
                }
            )

        total_feedback = len(feedback_items)
        avg_compatibility = round(sum(compatibility_values) / total_feedback, 1) if total_feedback else 0.0
        liked_rate = (liked_count / total_feedback) if total_feedback else 0.0
        meeting_rate = (meeting_count / total_feedback) if total_feedback else 0.0

        def _pct(value: Optional[float]) -> str:
            return f"{value:.1f}%" if value is not None else "н/д"

        table_rows = []
        for item in feedback_items:
            score_text = str(item["score"]) if item["score"] is not None else "-"
            table_rows.append(
                "<tr>"
                f"<td>{escape(item['created_at'])}</td>"
                f"<td>{escape(str(item['from']))}<br><small>{item['from_id']}</small></td>"
                f"<td>{escape(str(item['to']))}<br><small>{item['to_id']}</small></td>"
                f"<td>{self._feedback_label(item['liked'])}</td>"
                f"<td>{self._feedback_label(item['meeting'])}</td>"
                f"<td>{escape(score_text)}</td>"
                f"<td><b>{_pct(float(item['compatibility']))}</b></td>"
                f"<td>{_pct(item['questionnaire'])}</td>"
                f"<td>{_pct(item['tfidf'])}</td>"
                f"<td>{_pct(item['nlp'])}</td>"
                f"<td>{_pct(item['psychology'])}</td>"
                "</tr>"
            )

        score_rows = [(f"{score} звезд", float(count)) for score, count in score_distribution.items()]
        if no_score:
            score_rows.append(("без оценки", float(no_score)))
        compat_rows = [(label, float(value)) for label, value in compatibility_buckets.items()]
        liked_rows = [("понравилось", float(liked_count)), ("не понравилось", float(max(0, total_feedback - liked_count)))]
        meeting_rows = [("готовы снова", float(meeting_count)), ("остальные", float(max(0, total_feedback - meeting_count)))]

        generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        html = f"""<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <title>Отчет VK Dating Bot за {days} дней</title>
  <style>
    :root {{ color-scheme: light; --bg:#f5f7fb; --panel:#fff; --text:#111827; --muted:#6b7280; --line:#d8e0ea; --blue:#2563eb; --green:#059669; --orange:#d97706; }}
    body {{ margin:0; background:var(--bg); color:var(--text); font-family: Arial, Segoe UI, sans-serif; }}
    main {{ max-width:1180px; margin:0 auto; padding:38px; }}
    h1 {{ margin:0 0 6px; font-size:34px; }}
    h2 {{ margin:0 0 18px; font-size:22px; }}
    .muted, small {{ color:var(--muted); }}
    .grid {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; margin:26px 0; }}
    .card, .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:18px; box-shadow:0 8px 24px rgba(15,23,42,.06); }}
    .card {{ padding:18px; }}
    .card span {{ display:block; color:var(--muted); font-size:14px; }}
    .card b {{ display:block; margin-top:8px; font-size:28px; }}
    .panel {{ padding:22px; margin:18px 0; }}
    .charts {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    .bar-row {{ display:grid; grid-template-columns:130px 1fr 70px; align-items:center; gap:12px; margin:12px 0; }}
    .bar-track {{ height:16px; border-radius:999px; background:#e5eaf1; overflow:hidden; }}
    .bar-fill {{ height:100%; border-radius:999px; background:linear-gradient(90deg, var(--blue), #22c55e); }}
    .pie-wrap {{ display:grid; grid-template-columns:180px 1fr; gap:18px; align-items:center; min-height:180px; }}
    .pie {{ width:168px; height:168px; border-radius:50%; display:grid; place-items:center; position:relative; box-shadow:inset 0 0 0 1px rgba(15,23,42,.08); }}
    .pie::after {{ content:""; position:absolute; width:82px; height:82px; border-radius:50%; background:var(--panel); }}
    .pie strong {{ position:relative; z-index:1; font-size:24px; }}
    .pie-legend-row {{ display:grid; grid-template-columns:18px 1fr auto; gap:10px; align-items:center; margin:9px 0; }}
    .pie-legend-row span {{ width:18px; height:18px; border-radius:5px; }}
    .pie-legend-row b {{ font-weight:600; }}
    .pie-legend-row em {{ color:var(--muted); font-style:normal; }}
    .table-wrap {{ overflow-x:auto; }}
    table {{ width:100%; min-width:980px; border-collapse:collapse; font-size:14px; background:var(--panel); border-radius:18px; overflow:hidden; }}
    th, td {{ padding:12px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }}
    th {{ background:#edf3ff; font-size:12px; text-transform:uppercase; letter-spacing:.04em; color:#374151; }}
    tr:hover td {{ background:#f8fbff; }}
    .ok {{ color:var(--green); }}
    .warn {{ color:var(--orange); }}
  </style>
</head>
<body>
<main>
  <h1>Отчет VK Dating Bot за последние {days} дней</h1>
  <p class="muted">Сформирован: {escape(generated_at)}. Фидбеков в таблице: {total_feedback}.</p>

  <section class="grid">
    <div class="card"><span>Новые пользователи</span><b>{int(report.get('new_users', 0))}</b></div>
    <div class="card"><span>Лайки</span><b>{int(report.get('likes', 0))}</b></div>
    <div class="card"><span>Мэтчи</span><b>{int(report.get('matches', 0))}</b></div>
    <div class="card"><span>Средняя совместимость</span><b>{avg_compatibility:.1f}%</b></div>
    <div class="card"><span>Фидбеки</span><b>{total_feedback}</b></div>
    <div class="card"><span>Средняя оценка встреч</span><b>{report.get('avg_score', 0)}</b></div>
  </section>

  <section class="charts">
    <div class="panel">
      <h2>Оценки встреч</h2>
      {self._html_pie(score_rows)}
    </div>
    <div class="panel">
      <h2>Совместимость</h2>
      {self._html_pie(compat_rows)}
    </div>
    <div class="panel">
      <h2>Понравилось после встречи</h2>
      {self._html_pie(liked_rows)}
    </div>
    <div class="panel">
      <h2>Готовность встретиться снова</h2>
      {self._html_pie(meeting_rows)}
    </div>
  </section>

  <section class="panel">
    <h2>Фидбеки и проценты совместимости</h2>
    <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>Дата</th>
          <th>Кто</th>
          <th>Кому</th>
          <th>Понрав.</th>
          <th>Еще встреча</th>
          <th>Оценка</th>
          <th>Итог</th>
          <th>Анкета</th>
          <th>TF-IDF</th>
          <th>NLP</th>
          <th>Психо</th>
        </tr>
      </thead>
      <tbody>
        {''.join(table_rows) if table_rows else '<tr><td colspan="11" class="muted">Фидбеков за период нет</td></tr>'}
      </tbody>
    </table>
    </div>
  </section>
</main>
</body>
</html>
"""
        return html.encode("utf-8")

    # Обрабатывает команду или действие пользователя.
    def handle_admin_month_file_report(self, user_id: int) -> None:
        if not self._require_admin(user_id):
            return
        try:
            from datetime import datetime
            import zipfile

            html_data = self._build_monthly_feedback_report_html(days=30)
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            html_filename = f"vk_dating_month_report_{stamp}.html"
            filename = f"vk_dating_month_report_{stamp}.zip"
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(html_filename, html_data)
            data = zip_buffer.getvalue()

            fallback_dir = os.path.join("exports", "reports")
            os.makedirs(fallback_dir, exist_ok=True)
            fallback_path = os.path.join(fallback_dir, filename)
            with open(fallback_path, "wb") as f:
                f.write(data)

            attachment = self._upload_document_to_messages(
                user_id,
                data,
                filename=filename,
                mime_type="application/zip",
                title="Отчет VK Dating Bot за месяц.zip",
            )
        except Exception as e:
            self._log("Failed to build monthly file report for user=%s: %s", user_id, e)
            attachment = None

        if attachment:
            self.send_message(
                user_id,
                "Готово. Отчет за месяц можно скачать во вложении. Внутри ZIP лежит HTML-файл с визуализацией.",
                attachment=attachment,
            )
        else:
            saved_path = fallback_path if "fallback_path" in locals() else ""
            suffix = f"\nЛокальная копия сохранена: {saved_path}" if saved_path else ""
            self.send_message(user_id, "Не удалось загрузить файл отчета в VK." + suffix)

    def _seed_synthetic_nlp_files(self, examples_count: int = 40) -> Dict[str, int]:
        import csv
        from datetime import datetime

        from src.config import NLP_DATA_PATH
        from src.nlp_data_collector import NLP_FIELDNAMES
        from src.nlp_metrics import NLPMetricsTracker

        examples_count = max(1, min(int(examples_count), 300))
        output_dir = os.path.dirname(NLP_DATA_PATH)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        topics = [
            ("Люблю путешествия, спорт и активные выходные", "Обожаю поездки, прогулки и новые места"),
            ("Ценю уют, книги и спокойные вечера", "Люблю домашний формат, фильмы и долгие разговоры"),
            ("Много работаю, развиваю проекты и люблю цели", "Ценю амбиции, развитие и самостоятельность"),
            ("Мне нравятся вечеринки и большие компании", "Предпочитаю тишину, дом и камерное общение"),
            ("Хочу серьезные отношения и общие планы", "Ищу легкое общение без обязательств"),
        ]
        labels = ["positive", "positive", "neutral", "negative", "negative"]
        file_exists = os.path.exists(NLP_DATA_PATH)
        tracker = NLPMetricsTracker()
        batch_stamp = datetime.now().strftime("%Y%m%d%H%M%S")

        with open(NLP_DATA_PATH, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=NLP_FIELDNAMES)
            if not file_exists:
                writer.writeheader()

            for idx in range(examples_count):
                left, right = topics[idx % len(topics)]
                label = labels[idx % len(labels)]
                from_user_id = -910_000_000_000 - idx
                to_user_id = -920_000_000_000 - idx
                writer.writerow(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "from_user_id": from_user_id,
                        "to_user_id": to_user_id,
                        "text_left": f"{left}. synthetic_batch={batch_stamp}",
                        "text_right": f"{right}. synthetic_batch={batch_stamp}",
                        "label": label,
                    }
                )

                if label == "positive":
                    action = "like"
                    predicted_score = random.uniform(62, 92)
                elif label == "negative":
                    action = random.choice(["dislike", "block", "report"])
                    predicted_score = random.uniform(8, 38)
                else:
                    action = "skip"
                    predicted_score = random.uniform(42, 58)
                tracker.log_prediction(
                    viewer_id=from_user_id,
                    viewed_id=to_user_id,
                    predicted_score=predicted_score,
                    actual_action=action,
                    model_version="synthetic_demo",
                )

        return {"nlp_examples": examples_count, "nlp_metric_rows": examples_count}

    # Обрабатывает команду или действие пользователя.
    def handle_admin_stats_check(self, user_id: int) -> None:
        if not self._require_admin(user_id):
            return

        checks: List[Tuple[str, bool, str]] = []

        def run_check(name: str, func) -> object:
            try:
                result = func()
                checks.append((name, True, "готово"))
                return result
            except Exception as e:
                checks.append((name, False, str(e).splitlines()[0][:160]))
                return None

        snapshot = run_check("Таблицы статистики", lambda: self.db.get_stats_collection_snapshot(days=30))
        report = run_check("Месячный отчет БД", lambda: self.db.get_monthly_admin_report(days=30))
        funnel = run_check("Воронка событий", self.db.get_funnel_counts)

        def check_nlp_dataset() -> Dict[str, Any]:
            from src.nlp_data_collector import get_nlp_stats

            return get_nlp_stats()

        nlp_dataset = run_check("NLP-датасет", check_nlp_dataset)
        metrics = run_check("NLP-метрики", lambda: self._safe_nlp_metrics(None))

        def check_png() -> str:
            report_payload = report if isinstance(report, dict) else {}
            funnel_payload = funnel if isinstance(funnel, dict) else {}
            images = self._render_admin_stats_images(
                [
                    ("Пользователи", [("Новые", report_payload.get("new_users", 0)), ("Активные", report_payload.get("active_users", 0))]),
                    ("Подбор", [("Лайки", report_payload.get("likes", 0)), ("Мэтчи", report_payload.get("matches", 0))]),
                    ("Отзывы", [("Всего", report_payload.get("feedback_count", 0)), ("Оценка", report_payload.get("avg_score", 0))]),
                    ("NLP", [("Статус", "проверка"), ("Accuracy", "н/д")]),
                ],
                [
                    ("start", float(funnel_payload.get("start", 0))),
                    ("profile_complete", float(funnel_payload.get("profile_complete", 0))),
                    ("browse_started", float(funnel_payload.get("browse_started", 0))),
                    ("like_sent", float(funnel_payload.get("like_sent", 0))),
                    ("match", float(funnel_payload.get("match", 0))),
                ],
                [("positive", 1.0), ("neutral", 1.0), ("negative", 1.0)],
                [("accuracy all", 1.0)],
                [],
                [("успешные", 1.0), ("остальные", 1.0)],
                int(report_payload.get("reports", 0) or 0),
            )
            if len(images) < 3 or any(not image.startswith(b"\x89PNG") for image in images):
                raise RuntimeError("PNG pages missing")
            return f"{len(images)} картинки, {self._fmt_bytes(sum(len(image) for image in images))}"

        png_status = run_check("PNG-отчет", check_png)

        def check_month_file() -> str:
            import zipfile

            html = self._build_monthly_feedback_report_html(days=30)
            if not html.startswith(b"<!doctype html>"):
                raise RuntimeError("HTML signature missing")
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("check.html", html)
            data = buffer.getvalue()
            if not data.startswith(b"PK"):
                raise RuntimeError("ZIP signature missing")
            return f"HTML {self._fmt_bytes(len(html))}, ZIP {self._fmt_bytes(len(data))}"

        file_status = run_check("HTML/ZIP-файл", check_month_file)

        ok_count = sum(1 for _, ok, _ in checks if ok)
        all_ok = ok_count == len(checks)
        status_text = "всё работает" if all_ok else "есть ошибки"
        lines = [
            f"Проверка статистики: {ok_count}/{len(checks)} OK — {status_text}",
            "",
            "Источники данных:",
        ]
        for name, ok, detail in checks:
            status = "Готово" if ok else "Ошибка"
            lines.append(f"- {status}: {name}" + (f" — {detail}" if not ok else ""))

        if isinstance(snapshot, dict):
            lines.extend(
                [
                    "",
                    "Что собрано в базе:",
                    f"- Пользователи: {snapshot.get('users', 0)}",
                    f"- Анкеты: {snapshot.get('questionnaires', 0)}",
                    f"- Текстовые профили: {snapshot.get('text_profiles', 0)}",
                    f"- Фото: {snapshot.get('photos', 0)}",
                ]
            )
            lines.append(
                "\nЗа последние 30 дней:\n"
                f"- События: {snapshot.get('events_period', 0)}\n"
                f"- Лайки: {snapshot.get('likes_period', 0)}\n"
                f"- Фидбеки: {snapshot.get('feedback_period', 0)}\n"
                f"- Жалобы: {snapshot.get('reports_period', 0)}"
            )

        if isinstance(report, dict):
            likes = int(report.get("likes", 0))
            matches = int(report.get("matches", 0))
            feedback_count = int(report.get("feedback_count", 0))
            successful = int(report.get("successful_feedback", 0))
            match_rate = self._fmt_percent(matches / likes) if likes else "0.0%"
            success_rate = self._fmt_percent(successful / feedback_count) if feedback_count else "0.0%"
            lines.extend(
                [
                    "",
                    "Месячный отчет:",
                    f"- Новые пользователи: {report.get('new_users', 0)}",
                    f"- Активные пользователи: {report.get('active_users', 0)}",
                    f"- Лайки: {likes}",
                    f"- Мэтчи: {matches} ({match_rate} от лайков)",
                    f"- Фидбеки: {feedback_count}",
                    f"- Средняя оценка встреч: {report.get('avg_score', 0)}",
                    f"- Успешные встречи: {successful} ({success_rate})",
                    f"- Жалобы: {report.get('reports', 0)}",
                ]
            )
            top_events = report.get("top_events") or []
            if top_events:
                lines.append("Топ событий:")
                for row in top_events[:5]:
                    lines.append(f"- {row.get('event_name')}: {row.get('c')}")

        if isinstance(funnel, dict):
            lines.extend(
                [
                    "",
                    "Воронка:",
                    f"- Старт: {funnel.get('start', 0)}",
                    f"- Анкета готова: {funnel.get('profile_complete', 0)}",
                    f"- Начали смотреть анкеты: {funnel.get('browse_started', 0)}",
                    f"- Поставили лайк: {funnel.get('like_sent', 0)}",
                    f"- Получили мэтч: {funnel.get('match', 0)}",
                ]
            )

        if isinstance(metrics, dict):
            lines.extend(
                [
                    "",
                    "NLP-качество:",
                    f"- Предсказаний: {int(metrics.get('predictions_count', 0))}",
                    f"- Accuracy: {self._fmt_percent(float(metrics.get('accuracy', 0.0)))}",
                    f"- Precision: {self._fmt_percent(float(metrics.get('precision', 0.0)))}",
                    f"- Recall: {self._fmt_percent(float(metrics.get('recall', 0.0)))}",
                    f"- F1: {self._fmt_percent(float(metrics.get('f1', 0.0)))}",
                ]
            )

        if isinstance(nlp_dataset, dict):
            lines.extend(
                [
                    "",
                    "NLP-датасет:",
                    f"- Всего примеров: {int(nlp_dataset.get('total', 0))}",
                    f"- Положительные: {int(nlp_dataset.get('positive', 0))}",
                    f"- Нейтральные: {int(nlp_dataset.get('neutral', 0))}",
                    f"- Отрицательные: {int(nlp_dataset.get('negative', 0))}",
                    f"- Готов к обучению: {'да' if nlp_dataset.get('ready') else 'нет'}",
                ]
            )

        lines.extend(
            [
                "",
                "Рендер отчетов:",
                f"- PNG: {png_status if isinstance(png_status, str) else 'готово'}",
                f"- HTML/ZIP: {file_status if isinstance(file_status, str) else 'готово'}",
            ]
        )

        self.send_message(user_id, "\n".join(lines))

    # Обрабатывает команду или действие пользователя.
    def handle_admin_seed_stats(self, user_id: int) -> None:
        if not self._require_admin(user_id):
            return
        try:
            db_result = self.db.seed_synthetic_statistics(days=30, users_count=48, interactions_count=240)
            nlp_result = self._seed_synthetic_nlp_files(examples_count=100)
        except Exception as e:
            self._log("Failed to seed synthetic stats for user=%s: %s", user_id, e)
            self.send_message(user_id, f"Не удалось создать искусственную статистику: {str(e)[:160]}")
            return

        self.send_message(
            user_id,
            "Искусственная статистика обновлена.\n"
            f"Пакет: {db_result.get('batch_id')}\n"
            f"Заменено старых демо-пользователей: {db_result.get('cleared_users', 0)}\n"
            f"Новые демо-пользователи с именами: {db_result.get('users', 0)}\n"
            f"События: {db_result.get('events', 0)}\n"
            f"Лайки: {db_result.get('likes', 0)}\n"
            f"Взаимные лайки: {db_result.get('mutual_likes', 0)}\n"
            f"Фидбеки: {db_result.get('feedback', 0)}\n"
            f"Жалобы: {db_result.get('reports', 0)}\n"
            f"NLP-примеры: {nlp_result.get('nlp_examples', 0)}\n\n"
            "Теперь можно вызвать /admin_stats, /admin_month_file или /admin_stats_check.",
        )

    # Работает с фотографиями анкеты.
    def handle_photo_diagnostics(self, user_id: int) -> None:
        if not self._require_admin(user_id):
            return
        stats = self.db.get_photo_storage_stats()
        photos = self.db.get_user_photos(user_id)
        upload_status = "у администратора нет фото для тестовой выгрузки"
        attachment = None
        if photos:
            attachment = self._upload_photo_record_to_messages(user_id, photos[0])
            upload_status = "успешно" if attachment else "ошибка выгрузки через VK API"
        self.send_message(
            user_id,
            "Диагностика фото:\n"
            f"Фото в базе: {stats.get('photos_count', 0)}\n"
            f"Пользователей с фото: {stats.get('users_with_photos', 0)}\n"
            f"Размер в базе: {stats.get('total_bytes', 0)} байт\n"
            f"Тест выгрузки: {upload_status}",
        )
        if attachment:
            self.send_message(user_id, "Тестовое фото из базы:", attachment=attachment)

    # Обрабатывает команду или действие пользователя.
    def handle_message(self, event) -> None:
        user_id = self._event_user_id(event)
        if not user_id:
            self._log("Could not extract user_id from VK event: %s", event)
            return
        text = self._event_text(event).strip()
        lowered = text.lower()
        self._log(f"user={user_id} message='{text[:80]}'")

        if self._handle_create_profile(user_id, event):
            return
        if self._handle_browsing_action(user_id, lowered):
            return
        if self._handle_feedback(user_id, lowered):
            return
        
        # Handle feedback list selection
        state = self.user_states.get(user_id, {})
        if state.get("mode") == "feedback_list":
            if lowered == "назад":
                self.user_states.pop(user_id, None)
                self.send_message(user_id, "Выберите действие.", keyboard=self._profile_keyboard())
                return

            if lowered in {"◀", "◁", "<"}:
                current = int(state.get("feedback_page", 0))
                state["feedback_page"] = max(0, current - 1)
                self.user_states[user_id] = state
                self._render_feedback_list_page(user_id)
                return

            if lowered in {"▶", "▷", ">"}:
                current = int(state.get("feedback_page", 0))
                state["feedback_page"] = current + 1
                self.user_states[user_id] = state
                self._render_feedback_list_page(user_id)
                return

            name_to_uid = state.get("name_to_uid", {})
            if isinstance(name_to_uid, dict):
                uid = name_to_uid.get(lowered)
                if isinstance(uid, int):
                    self._start_feedback(user_id, uid)
                    return

        if lowered in {"начать", "start"}:
            self.handle_start(user_id)
            return
        if lowered == "создать анкету":
            self._start_create_profile(user_id, reset=False)
            return
        if lowered == "перезаполнить анкету":
            self._start_create_profile(user_id, reset=True)
            return
        if lowered == "показать свою анкету":
            self.handle_show_profile(user_id)
            return
        if lowered in {"подбор анкет", "смотреть анкеты"}:
            self.handle_match(user_id)
            return
        if lowered == "кто лайкнул":
            self.handle_incoming_likes(user_id)
            return
        if lowered == "мэтчи":
            self.handle_matches(user_id)
            return
        if lowered == "оставить отзыв":
            self.handle_feedback_list(user_id)
            return
        if lowered == "/admin_reports":
            self.handle_admin_reports(user_id)
            return
        if lowered == "/admin_funnel":
            self.handle_admin_funnel(user_id)
            return
        if lowered in {"/admin_stats", "/admin_nlp"}:
            self.handle_admin_stats(user_id)
            return
        if lowered in {"/admin_month", "/admin_report_month"}:
            self.handle_admin_month_report(user_id)
            return
        if lowered in {"/admin_month_file", "/admin_report_file"}:
            self.handle_admin_month_file_report(user_id)
            return
        if lowered in {"/admin_stats_check", "/admin_check_stats"}:
            self.handle_admin_stats_check(user_id)
            return
        if lowered in {"/admin_seed_stats", "/admin_fake_stats"}:
            self.handle_admin_seed_stats(user_id)
            return
        if lowered in {"/admin_photo_check", "/photo_check"}:
            self.handle_photo_diagnostics(user_id)
            return

        profile = self.db.get_user_profile(user_id)
        if self._is_profile_complete(profile):
            self.send_message(user_id, "Выберите действие из меню профиля.", keyboard=self._profile_keyboard())
        else:
            self.send_message(user_id, "Для начала нажмите 'Начать'.", keyboard=self._start_keyboard())

    # Запускает основной цикл обработки событий.
    def run(self) -> None:
        if self.bot_longpoll:
            for event in self.bot_longpoll.listen():
                if event.type == VkBotEventType.MESSAGE_NEW:
                    try:
                        self.handle_message(event)
                    except Exception as e:
                        self._log(f"ОШИБКА обработки сообщения: {str(e)}")
            return

        for event in self.longpoll.listen():
            if event.type == VkEventType.MESSAGE_NEW and event.to_me:
                try:
                    self.handle_message(event)
                except Exception as e:
                    self._log(f"ОШИБКА обработки сообщения: {str(e)}")
                    
