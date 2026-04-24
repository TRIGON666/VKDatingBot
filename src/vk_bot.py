from __future__ import annotations

import random
import io
import urllib.request
import logging
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import requests
import vk_api
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

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger("vk_bot")


class VKCompatibilityBot:
    DAILY_LIKE_LIMIT = 50
    DAILY_DISLIKE_LIMIT = 200
    FEEDBACK_PAGE_SIZE = 10
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

    def __init__(self, token: str, db: Database) -> None:
        self.vk_session = vk_api.VkApi(token=token)
        self.vk = self.vk_session.get_api()
        self.longpoll = VkLongPoll(self.vk_session)
        self.db = db
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

    def _normalize_city_raw(self, city: str) -> str:
        value = city.lower().replace("ё", "е").strip()
        value = re.sub(r"[.,\-_/]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        value = re.sub(r"\b(г|гор|город|обл|область|край|респ|республика|район|р н|рн)\b", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

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

        # Нахождение токенов возможны, что найдёно в строках типа "санкт петербург центр" или "москва югвао".
        for alias, canonical in self._city_alias_items:
            if len(alias) >= 4 and (value in alias or alias in value):
                self._city_norm_cache[city] = canonical
                return canonical

        # Нечёткое сравнение прихватывает топографические ошибки и расницы.
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

    def _start_keyboard(self) -> VkKeyboard:
        kb = VkKeyboard(one_time=False)
        kb.add_button("Начать", color=VkKeyboardColor.POSITIVE)
        return kb

    def _create_keyboard(self) -> VkKeyboard:
        kb = VkKeyboard(one_time=False)
        kb.add_button("Создать анкету", color=VkKeyboardColor.PRIMARY)
        return kb

    def _profile_keyboard(self) -> VkKeyboard:
        kb = VkKeyboard(one_time=False)
        kb.add_button("Показать свою анкету", color=VkKeyboardColor.SECONDARY)
        kb.add_line()
        kb.add_button("Смотреть анкеты", color=VkKeyboardColor.POSITIVE)
        kb.add_button("Кто лайкнул", color=VkKeyboardColor.PRIMARY)
        kb.add_line()
        kb.add_button("Мэтчи", color=VkKeyboardColor.POSITIVE)
        kb.add_button("Оставить отзыв", color=VkKeyboardColor.SECONDARY)
        kb.add_line()
        kb.add_button("Перезаполнить анкету", color=VkKeyboardColor.SECONDARY)
        return kb

    def _browse_keyboard(self) -> VkKeyboard:
        kb = VkKeyboard(one_time=False)
        kb.add_button("❤", color=VkKeyboardColor.POSITIVE)
        kb.add_button("👎", color=VkKeyboardColor.NEGATIVE)
        kb.add_line()
        kb.add_button("⚠ Жалоба", color=VkKeyboardColor.SECONDARY)
        kb.add_button("🚫 Блок", color=VkKeyboardColor.SECONDARY)
        kb.add_line()
        kb.add_button("Стоп", color=VkKeyboardColor.SECONDARY)
        return kb

    def _cancel_keyboard(self) -> VkKeyboard:
        kb = VkKeyboard(one_time=False)
        kb.add_button("Отмена", color=VkKeyboardColor.NEGATIVE)
        return kb

    def _gender_keyboard(self) -> VkKeyboard:
        kb = VkKeyboard(one_time=False)
        kb.add_button("Мужской", color=VkKeyboardColor.PRIMARY)
        kb.add_button("Женский", color=VkKeyboardColor.PRIMARY)
        kb.add_line()
        kb.add_button("Отмена", color=VkKeyboardColor.NEGATIVE)
        return kb

    def _question_keyboard(self, question_idx: int) -> VkKeyboard:
        q = QUESTIONS[question_idx]
        kb = VkKeyboard(one_time=False)
        for idx, option in enumerate(q.options):
            kb.add_button(option, color=VkKeyboardColor.PRIMARY)
            if idx % 2 == 1 and idx != len(q.options) - 1:
                kb.add_line()
        kb.add_line()
        kb.add_button("Свой вариант", color=VkKeyboardColor.SECONDARY)
        kb.add_line()
        kb.add_button("Отмена", color=VkKeyboardColor.NEGATIVE)
        return kb

    def _photo_keyboard(self) -> VkKeyboard:
        kb = VkKeyboard(one_time=False)
        kb.add_button("Готово", color=VkKeyboardColor.POSITIVE)
        kb.add_button("Отмена", color=VkKeyboardColor.NEGATIVE)
        return kb

    def _yes_no_keyboard(self) -> VkKeyboard:
        kb = VkKeyboard(one_time=False)
        kb.add_button("Да", color=VkKeyboardColor.POSITIVE)
        kb.add_button("Нет", color=VkKeyboardColor.NEGATIVE)
        return kb

    def _rating_keyboard(self) -> VkKeyboard:
        kb = VkKeyboard(one_time=False)
        kb.add_button("1", color=VkKeyboardColor.NEGATIVE)
        kb.add_button("2", color=VkKeyboardColor.SECONDARY)
        kb.add_button("3", color=VkKeyboardColor.SECONDARY)
        kb.add_button("4", color=VkKeyboardColor.SECONDARY)
        kb.add_button("5", color=VkKeyboardColor.POSITIVE)
        kb.add_line()
        kb.add_button("Отмена", color=VkKeyboardColor.NEGATIVE)
        return kb

    def _psychology_keyboard(self) -> VkKeyboard:
        """Новая клавиатура формы опроса с пюикассу 1-5."""
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

    def _normalize_photo_attachment(self, raw: str) -> Optional[str]:
        value = str(raw).strip()
        if not value:
            return None
        if value.startswith("photo"):
            return value
        if "_" in value:
            return f"photo{value}"
        return None

    def _extract_photo_attachments(self, event) -> List[str]:
        attachments: List[str] = []
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

    def _extract_photo_attachments_by_message_id(self, message_id: Optional[int]) -> List[str]:
        if not message_id:
            return []
        try:
            response = self.vk.messages.getById(message_ids=message_id)
            items = response.get("items", [])
            if not items:
                return []
            result: List[str] = []
            for item in items[0].get("attachments", []):
                if item.get("type") != "photo":
                    continue
                photo = item.get("photo", {})
                owner_id = photo.get("owner_id")
                photo_id = photo.get("id")
                access_key = photo.get("access_key")
                if owner_id is None or photo_id is None:
                    continue
                result.append(f"photo{owner_id}_{photo_id}_{access_key}" if access_key else f"photo{owner_id}_{photo_id}")
            return result
        except (ValueError, KeyError, IndexError):
            return []
        except Exception as e:
            self._log(f"Ошибка извлечения вложений с фото: {e}")
            return []

    def _rehost_photo_for_messages(self, user_id: int, raw_attachment: str) -> Optional[str]:
        attachment = self._normalize_photo_attachment(raw_attachment)
        if not attachment:
            return None
        try:
            photo_info = self.vk.photos.getById(photos=attachment)
            sizes = photo_info[0].get("sizes", []) if photo_info else []
            if not sizes:
                return attachment
            best = max(sizes, key=lambda s: int(s.get("width", 0)) * int(s.get("height", 0)))
            photo_url = best.get("url")
            if not photo_url:
                return attachment
            with urllib.request.urlopen(photo_url, timeout=15) as resp:
                image_bytes = resp.read()
            upload_server = self.vk.photos.getMessagesUploadServer(peer_id=user_id)
            upload_url = upload_server.get("upload_url")
            if not upload_url:
                return attachment
            uploaded = requests.post(upload_url, files={"photo": ("profile.jpg", image_bytes, "image/jpeg")}, timeout=20)
            if uploaded.status_code != 200:
                return attachment
            payload = uploaded.json()
            saved = self.vk.photos.saveMessagesPhoto(photo=payload.get("photo"), server=payload.get("server"), hash=payload.get("hash"))
            if not saved:
                return attachment
            p = saved[0]
            owner_id, photo_id, access_key = p.get("owner_id"), p.get("id"), p.get("access_key")
            if owner_id is None or photo_id is None:
                return attachment
            return f"photo{owner_id}_{photo_id}_{access_key}" if access_key else f"photo{owner_id}_{photo_id}"
        except (urllib.error.URLError, requests.RequestException, IndexError, KeyError):
            return attachment
        except Exception as e:
            self._log(f"Ошибка пере-загрузки фото: {e}")
            return attachment

    def _download_photo_blob(self, raw_attachment: str) -> Optional[Tuple[bytes, str, str]]:
        attachment = self._normalize_photo_attachment(raw_attachment)
        if not attachment:
            return None
        try:
            photo_info = self.vk.photos.getById(photos=attachment)
            if not photo_info:
                return None
            sizes = photo_info[0].get("sizes", [])
            if not sizes:
                return None
            best = max(sizes, key=lambda s: int(s.get("width", 0)) * int(s.get("height", 0)))
            photo_url = best.get("url")
            if not photo_url:
                return None
            with urllib.request.urlopen(photo_url, timeout=20) as resp:
                data = resp.read()
                content_type = str(resp.headers.get_content_type() or "image/jpeg")
            ext = "jpg"
            if "png" in content_type:
                ext = "png"
            elif "webp" in content_type:
                ext = "webp"
            elif "gif" in content_type:
                ext = "gif"
            filename = f"photo.{ext}"
            return data, content_type, filename
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError):
            return None
        except Exception as e:
            self._log(f"Ошибка загрузки фото-данных: {e}")
            return None

    def _download_photo_blobs_by_message_id(self, message_id: Optional[int]) -> List[Tuple[bytes, str, str]]:
        if not message_id:
            return []
        try:
            response = self.vk.messages.getById(message_ids=message_id)
            items = response.get("items", [])
            if not items:
                return []
            result: List[Tuple[bytes, str, str]] = []
            for item in items[0].get("attachments", []):
                if item.get("type") != "photo":
                    continue
                photo = item.get("photo", {})
                sizes = photo.get("sizes", [])
                if not sizes:
                    continue
                best = max(sizes, key=lambda s: int(s.get("width", 0)) * int(s.get("height", 0)))
                photo_url = best.get("url")
                if not photo_url:
                    continue
                with urllib.request.urlopen(photo_url, timeout=20) as resp:
                    data = resp.read()
                    content_type = str(resp.headers.get_content_type() or "image/jpeg")
                ext = "jpg"
                if "png" in content_type:
                    ext = "png"
                elif "webp" in content_type:
                    ext = "webp"
                elif "gif" in content_type:
                    ext = "gif"
                result.append((data, content_type, f"photo.{ext}"))
            return result
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError):
            return []
        except Exception as e:
            self._log(f"Ошибка загрузки фото из сообщения {message_id}: {e}")
            return []

    def _upload_photo_record_to_messages(self, user_id: int, photo: PhotoRecord) -> Optional[str]:
        try:
            upload_server = self.vk.photos.getMessagesUploadServer(peer_id=user_id)
            upload_url = upload_server.get("upload_url")
            if not upload_url:
                return None
            uploaded = requests.post(
                upload_url,
                files={
                    "photo": (
                        photo.get("filename", "photo.jpg"),
                        io.BytesIO(photo["photo_data"]),
                        photo.get("mime_type", "image/jpeg"),
                    )
                },
                timeout=30,
            )
            if uploaded.status_code != 200:
                return None
            payload = uploaded.json()
            saved = self.vk.photos.saveMessagesPhoto(
                photo=payload.get("photo"),
                server=payload.get("server"),
                hash=payload.get("hash"),
            )
            if not saved:
                return None
            p = saved[0]
            owner_id = p.get("owner_id")
            photo_id = p.get("id")
            access_key = p.get("access_key")
            if owner_id is None or photo_id is None:
                return None
            return f"photo{owner_id}_{photo_id}_{access_key}" if access_key else f"photo{owner_id}_{photo_id}"
        except Exception:
            return None

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
        if step == "question_custom":
            idx = int(state.get("custom_question_idx", 0))
            if idx < 0 or idx >= len(QUESTIONS):
                state["step"] = "questions"
                state["question_idx"] = 0
                state.pop("custom_question_idx", None)
                self.user_states[user_id] = state
                self.db.save_draft(user_id, state)
                self._next_create_step(user_id)
                return
            q = QUESTIONS[idx]
            self.send_message(
                user_id,
                f"✍️ Введите свой вариант для '{q.text}' (2-40 символов).",
                keyboard=self._cancel_keyboard(),
            )
            return
        if step == "about":
            self.send_message(
                user_id,
                "📝 Шаг {}/{}: расскажите о себе (можно '-' для пустого описания).".format(
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

    def _handle_create_profile(self, user_id: int, event) -> bool:
        state = self.user_states.get(user_id)
        if not state or state.get("mode") != "create_profile":
            return False
        text = (event.text or "").strip()
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
            self.user_states[user_id] = state
            self.db.save_draft(user_id, state)
            self._next_create_step(user_id)
            return True

        if step == "gender":
            if text not in {"Мужской", "Женский"}:
                self.send_message(user_id, "Выберите ваш пол кнопкой ниже.", keyboard=self._gender_keyboard())
                return True
            answers["gender"] = text
            state["step"] = "age"
            self.user_states[user_id] = state
            self.db.save_draft(user_id, state)
            self._next_create_step(user_id)
            return True

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
            self.user_states[user_id] = state
            self.db.save_draft(user_id, state)
            self._next_create_step(user_id)
            return True

        if step == "city":
            city = text.strip()
            if len(city) < 2:
                self.send_message(user_id, "Город слишком короткий.", keyboard=self._cancel_keyboard())
                return True
            answers["city"] = self._normalize_city(city)
            state["step"] = "search_gender"
            self.user_states[user_id] = state
            self.db.save_draft(user_id, state)
            self._next_create_step(user_id)
            return True

        if step == "search_gender":
            if text not in {"Мужской", "Женский"}:
                self.send_message(user_id, "Выберите, кого хотите найти, кнопкой ниже.", keyboard=self._gender_keyboard())
                return True
            answers["search_gender"] = text
            state["step"] = "questions"
            state["question_idx"] = 0
            self.user_states[user_id] = state
            self.db.save_draft(user_id, state)
            self._next_create_step(user_id)
            return True

        if step == "questions":
            idx = int(state.get("question_idx", 0))
            if idx < 0 or idx >= len(QUESTIONS):
                state["step"] = "about"
                state.pop("question_idx", None)
                self.user_states[user_id] = state
                self.db.save_draft(user_id, state)
                self._next_create_step(user_id)
                return True
            q = QUESTIONS[idx]

            option_map = {str(opt).strip().lower(): opt for opt in q.options}
            normalized_text = text.strip().lower()
            if normalized_text == "свой вариант":
                state["step"] = "question_custom"
                state["custom_question_idx"] = idx
                self.user_states[user_id] = state
                self.db.save_draft(user_id, state)
                self._next_create_step(user_id)
                return True

            selected = option_map.get(normalized_text)
            if selected is None:
                self.send_message(
                    user_id,
                    "Выберите вариант кнопкой или нажмите 'Свой вариант'.",
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
            self.user_states[user_id] = state
            self.db.save_draft(user_id, state)
            self._next_create_step(user_id)
            return True

        if step == "question_custom":
            idx = int(state.get("custom_question_idx", 0))
            if idx < 0 or idx >= len(QUESTIONS):
                state["step"] = "questions"
                state["question_idx"] = 0
                state.pop("custom_question_idx", None)
                self.user_states[user_id] = state
                self.db.save_draft(user_id, state)
                self._next_create_step(user_id)
                return True

            custom_value = text.strip().lower()
            if len(custom_value) < 2 or len(custom_value) > 40:
                self.send_message(
                    user_id,
                    "Свой вариант должен быть длиной от 2 до 40 символов.",
                    keyboard=self._cancel_keyboard(),
                )
                return True

            q = QUESTIONS[idx]
            answers[q.key] = custom_value
            idx += 1
            state.pop("custom_question_idx", None)
            if idx >= len(QUESTIONS):
                state["step"] = "about"
                state.pop("question_idx", None)
            else:
                state["step"] = "questions"
                state["question_idx"] = idx
            self.user_states[user_id] = state
            self.db.save_draft(user_id, state)
            self._next_create_step(user_id)
            return True

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
                self.user_states[user_id] = state
                self.db.save_draft(user_id, state)
                self._next_create_step(user_id)
                return True
            psych_answers = state.get("psychology_answers", {})
            psych_answers[q_id] = answer
            
            idx += 1
            state["psychology_question_idx"] = idx
            state["psychology_answers"] = psych_answers
            
            if idx >= len(PSYCHOLOGY_QUESTIONS):
                self.user_states[user_id] = state
                self.db.save_draft(user_id, state)
                self._finish_create_profile(user_id)
            else:
                self.user_states[user_id] = state
                self.db.save_draft(user_id, state)
                self._next_create_step(user_id)
            
            return True

        if step == "about":
            state["about_text"] = "" if text == "-" else text
            state["step"] = "photos"
            self.user_states[user_id] = state
            self.db.save_draft(user_id, state)
            self._next_create_step(user_id)
            return True

        if step == "photos":
            if lowered == "готово":
                total = len(self.db.get_user_photos(user_id))
                if total < 1:
                    self.send_message(user_id, "Нужно загрузить хотя бы одно фото перед завершением.", keyboard=self._photo_keyboard())
                    return True
                # Переход к анонимному тесту
                state = self.user_states.get(user_id, {})
                state["step"] = "psychology_questions"
                state["psychology_question_idx"] = 0
                state["psychology_answers"] = {}
                state["psychology_intro_shown"] = False
                self.user_states[user_id] = state
                self.db.save_draft(user_id, state)
                self._next_create_step(user_id)
                return True
            message_id = getattr(event, "message_id", None)
            blobs = self._download_photo_blobs_by_message_id(message_id)
            if blobs:
                ready = [{"photo_data": b[0], "mime_type": b[1], "filename": b[2]} for b in blobs]
                total = self.db.add_user_photos(user_id, ready)
                self.db.log_event(user_id, "photo_uploaded")
                self._log(f"user={user_id} uploaded photos count={len(ready)} total={total}")
                self.send_message(
                    user_id,
                    f"Фото добавлены: +{len(ready)}. Сейчас в анкете: {total}.",
                    keyboard=self._photo_keyboard(),
                )
                return True
            photos = self._extract_photo_attachments(event)
            if not photos:
                photos = self._extract_photo_attachments_by_message_id(message_id)
            photos = [p for p in (self._normalize_photo_attachment(v) for v in photos) if p]
            if not photos:
                self.send_message(
                    user_id,
                    "Я не вижу фото во вложении. Отправьте фото или нажмите 'Готово'.",
                    keyboard=self._photo_keyboard(),
                )
                return True
            blobs = [self._download_photo_blob(p) for p in photos]
            ready = [
                {"photo_data": b[0], "mime_type": b[1], "filename": b[2]}
                for b in blobs
                if b is not None
            ]
            if not ready:
                self.send_message(
                    user_id,
                    "Не удалось обработать это фото. Попробуйте другое изображение.",
                    keyboard=self._photo_keyboard(),
                )
                return True
            total = self.db.add_user_photos(user_id, ready)
            self.db.log_event(user_id, "photo_uploaded")
            self._log(f"user={user_id} uploaded photos count={len(ready)} total={total}")
            self.send_message(
                user_id,
                f"Фото добавлены: +{len(ready)}. Сейчас в анкете: {total}.",
                keyboard=self._photo_keyboard(),
            )
            return True
        return False

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
            if item.key == "age_group":
                continue
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
            if item.key == "age_group":
                continue
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
                "text_score": float(first.text_score),
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

        # Если город пользователя программирован, искать только в том же нормализованном городе.
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
                "text_score": float(r.text_score),
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
                        "text_score": float(r.text_score),
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
                    "text_score": 0.0,
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
        self.send_message(user_id, "Новые симпатии. Можно сразу ответить взаимностью кнопкой ❤.", keyboard=self._browse_keyboard())
        self._show_next_candidate(user_id)

    def handle_matches(self, user_id: int) -> None:
        match_ids = self.db.get_mutual_match_user_ids(user_id)
        if not match_ids:
            self.send_message(user_id, "Пока нет взаимных симпатий.", keyboard=self._profile_keyboard())
            return
        for uid in match_ids[:10]:
            other = self.db.get_user_profile(uid)
            nick = self._get_nick(other)
            self._send_profile_preview(user_id, uid, f"Ваш мэтч: [id{uid}|{nick}]")

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
            f"Выберите матч для отзыва (страница {page + 1}/{max_page + 1}):",
            keyboard=kb,
        )

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
                state["step"] = "liked"
                self.user_states[user_id] = state
                self.send_message(
                    user_id,
                    f"Встреча с {other_nick} вам понравилась?",
                    keyboard=self._yes_no_keyboard(),
                )
                return True
            elif lowered == "нет":
                state["liked"] = 0
                state["meeting_agree"] = 0

                self.db.save_feedback(
                    user_id,
                    other_user_id,
                    state["liked"],
                    state["meeting_agree"],
                    None,
                )

                self.user_states.pop(user_id, None)
                self.send_message(
                    user_id,
                    "Отзыв сохранён. Дополнительные вопросы не задаю, так как встреча не состоялась.",
                    keyboard=self._profile_keyboard(),
                )
                self._send_feedback_stats(user_id)
                self._log(f"user={user_id} feedback saved for {other_user_id} meeting_not_happened=1")
                return True
            else:
                self.send_message(user_id, "Выберите 'Да' или 'Нет'.", keyboard=self._yes_no_keyboard())
                return True
        
        if step == "liked":
            if lowered == "да":
                state["liked"] = 1
                state["step"] = "agree"
                self.user_states[user_id] = state
                self.send_message(
                    user_id,
                    f"Вы согласны встречаться с {other_nick} снова?",
                    keyboard=self._yes_no_keyboard(),
                )
                return True
            elif lowered == "нет":
                state["liked"] = 0
                state["step"] = "rating"
                self.user_states[user_id] = state
                self.send_message(
                    user_id,
                    f"Оцените встречу с {other_nick} от 1 до 5 звёзд:",
                    keyboard=self._rating_keyboard(),
                )
                return True
            else:
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
            
            state["step"] = "rating"
            self.user_states[user_id] = state
            self.send_message(
                user_id,
                f"Оцените встречу с {other_nick} от 1 до 5 звёзд:",
                keyboard=self._rating_keyboard(),
            )
            return True
        
        if step == "rating":
            if lowered in {"1", "2", "3", "4", "5"}:
                score = int(lowered)
                state["user_score"] = score
                
                self.db.save_feedback(
                    user_id,
                    other_user_id,
                    state["liked"],
                    state["meeting_agree"],
                    score,
                )
                
                self.user_states.pop(user_id, None)
                self.send_message(
                    user_id,
                    f"Спасибо за отзыв! 🙏\n\nОценка: {'⭐' * score}",
                    keyboard=self._profile_keyboard(),
                )
                self._send_feedback_stats(user_id)
                self._log(f"user={user_id} feedback saved for {other_user_id} score={score} liked={state['liked']} meeting={state['meeting_agree']}")
                return True
            else:
                self.send_message(user_id, "Выберите оценку от 1 до 5.", keyboard=self._rating_keyboard())
                return True
        
        return False

    def handle_admin_reports(self, user_id: int) -> None:
        reports = self.db.get_reports(limit=10)
        if not reports:
            self.send_message(user_id, "Жалоб нет.")
            return
        lines = ["Последние жалобы:"]
        for r in reports:
            lines.append(f"- {r['from_user_id']} -> {r['to_user_id']} ({r['reason']})")
        self.send_message(user_id, "\n".join(lines))

    def handle_admin_funnel(self, user_id: int) -> None:
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

    def handle_message(self, event) -> None:
        user_id = event.user_id
        text = (event.text or "").strip()
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

        profile = self.db.get_user_profile(user_id)
        if self._is_profile_complete(profile):
            self.send_message(user_id, "Выберите действие из меню профиля.", keyboard=self._profile_keyboard())
        else:
            self.send_message(user_id, "Для начала нажмите 'Начать'.", keyboard=self._start_keyboard())

    def run(self) -> None:
        for event in self.longpoll.listen():
            if event.type == VkEventType.MESSAGE_NEW and event.to_me:
                try:
                    self.handle_message(event)
                except Exception as e:
                    self._log(f"ОШИБКА обработки сообщения: {str(e)}")
