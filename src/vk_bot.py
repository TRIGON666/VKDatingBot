from __future__ import annotations

import io
import logging
import random
import re
import os
import sys
from collections.abc import Mapping
from difflib import SequenceMatcher
from html import unescape
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
import vk_api
from vk_api.bot_longpoll import VkBotEventType, VkBotLongPoll
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.longpoll import VkEventType, VkLongPoll

from src.analytics import collect_feedback_stats
from src.database import Database, PhotoRecord, UserProfile
from src.matching import rank_matches
from src.questionnaire import QUESTIONS
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

    # Работает с анкетой или профилем пользователя.
    def _is_profile_complete(self, profile: Optional[UserProfile]) -> bool:
        if not profile:
            return False
        required = {"age", "gender", "search_gender", "nickname"}
        if not required.issubset(set(profile.questionnaire.keys())):
            return False
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
        total_steps = 7 + len(QUESTIONS) + len(PSYCHOLOGY_QUESTIONS)
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
        total_steps = 7 + len(QUESTIONS) + len(PSYCHOLOGY_QUESTIONS)
        
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
            self.send_message(
                user_id,
                "🔎 Шаг 5/{}: кого вы хотите найти?".format(total_steps),
                keyboard=self._gender_keyboard(),
            )
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
                f"🎯 Шаг {6 + idx}/{total_steps}: {q.text}",
                keyboard=self._question_keyboard(idx),
            )
            return
        if step == "about":
            self.send_message(
                user_id,
                "📝 Шаг {}/{}: расскажите о себе. Ваши увлечения, чем занимаетесь в свободное время, учитесь/работаете?".format(
                    6 + len(QUESTIONS), total_steps
                ),
                keyboard=self._cancel_keyboard(),
            )
            return
        if step == "photos":
            self.send_message(
                user_id,
                "📸 Шаг {}/{}: загрузите минимум 1 фото.\nПосле отправки нажмите 'Готово'.".format(
                    7 + len(QUESTIONS), total_steps
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
            step_num = 8 + len(QUESTIONS) + idx
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
            state["step"] = "search_gender"
            return self._continue_create_step(user_id, state)

        if step == "search_gender":
            if text not in {"Мужской", "Женский"}:
                self.send_message(user_id, "Выберите, кого хотите найти, кнопкой ниже.", keyboard=self._gender_keyboard())
                return True
            answers["search_gender"] = text
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
        def _to_int(value: object, default: int = 0) -> int:
            try:
                return int(str(value).strip())
            except (TypeError, ValueError):
                return default

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
            and str(c.questionnaire.get("search_gender", "")).strip().lower() == base_gender
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
        if lowered in {"/admin_month", "/admin_report_month"}:
            self.handle_admin_month_report(user_id)
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
                    