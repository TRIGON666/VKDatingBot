from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, TypedDict

import psycopg
from psycopg.rows import dict_row


class PhotoRecord(TypedDict):
    photo_id: int
    photo_data: bytes
    mime_type: str
    filename: str


@dataclass
class UserProfile:
    user_id: int
    questionnaire: Dict[str, str]
    about_text: str
    photos: List[PhotoRecord]


class Database:
    """PostgreSQL storage used by the bot and NLP utilities."""

    # Инициализирует объект и сохраняет нужные зависимости.
    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("DATABASE_URL is required for PostgreSQL connection")
        self.database_url = database_url
        self._init_schema()

    # Открывает подключение к базе данных.
    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.database_url, row_factory=dict_row)

    # Выполняет SQL-запрос с фиксацией изменений.
    def _execute(self, query: str, params: Sequence[Any] = ()) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                conn.commit()

    # Выполняет SELECT-запрос и возвращает строки.
    def _fetch_all(self, query: str, params: Sequence[Any] = ()) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

    # Выполняет SELECT-запрос и возвращает строки.
    def _fetch_one(self, query: str, params: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchone()

    # Создает и обновляет таблицы базы данных.
    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS users (
                        user_id BIGINT PRIMARY KEY,
                        registered_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS questionnaire_answers (
                        user_id BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                        answers_json JSONB NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS text_profiles (
                        user_id BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                        about_text TEXT NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS feedback (
                        id BIGSERIAL PRIMARY KEY,
                        from_user_id BIGINT NOT NULL,
                        to_user_id BIGINT NOT NULL,
                        liked INTEGER DEFAULT 0,
                        meeting_agree INTEGER DEFAULT 0,
                        user_score INTEGER,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS user_photos (
                        photo_id BIGSERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
                        photo_data BYTEA NOT NULL,
                        mime_type TEXT NOT NULL DEFAULT 'image/jpeg',
                        filename TEXT NOT NULL DEFAULT 'photo.jpg',
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS likes (
                        id BIGSERIAL PRIMARY KEY,
                        from_user_id BIGINT NOT NULL,
                        to_user_id BIGINT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE (from_user_id, to_user_id)
                    );

                    CREATE TABLE IF NOT EXISTS dislikes (
                        id BIGSERIAL PRIMARY KEY,
                        from_user_id BIGINT NOT NULL,
                        to_user_id BIGINT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE (from_user_id, to_user_id)
                    );

                    CREATE TABLE IF NOT EXISTS blocks (
                        id BIGSERIAL PRIMARY KEY,
                        from_user_id BIGINT NOT NULL,
                        to_user_id BIGINT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        UNIQUE (from_user_id, to_user_id)
                    );

                    CREATE TABLE IF NOT EXISTS reports (
                        id BIGSERIAL PRIMARY KEY,
                        from_user_id BIGINT NOT NULL,
                        to_user_id BIGINT NOT NULL,
                        reason TEXT NOT NULL,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS user_drafts (
                        user_id BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                        draft_json JSONB NOT NULL,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS events (
                        id BIGSERIAL PRIMARY KEY,
                        user_id BIGINT NOT NULL,
                        event_name TEXT NOT NULL,
                        meta_json JSONB,
                        created_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE TABLE IF NOT EXISTS psychology_profile (
                        user_id BIGINT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
                        answers_json JSONB NOT NULL,
                        scores_json JSONB,
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    DELETE FROM feedback f1
                    USING feedback f2
                    WHERE f1.from_user_id = f2.from_user_id
                      AND f1.to_user_id = f2.to_user_id
                      AND f1.id < f2.id;

                    CREATE UNIQUE INDEX IF NOT EXISTS feedback_from_to_uidx
                        ON feedback (from_user_id, to_user_id);
                    CREATE INDEX IF NOT EXISTS likes_to_user_created_idx
                        ON likes (to_user_id, created_at DESC);
                    CREATE INDEX IF NOT EXISTS dislikes_from_user_created_idx
                        ON dislikes (from_user_id, created_at DESC);
                    CREATE INDEX IF NOT EXISTS blocks_from_user_created_idx
                        ON blocks (from_user_id, created_at DESC);
                    CREATE INDEX IF NOT EXISTS events_user_event_created_idx
                        ON events (user_id, event_name, created_at DESC);
                    CREATE INDEX IF NOT EXISTS reports_to_user_created_idx
                        ON reports (to_user_id, created_at DESC);

                    DO $$
                    BEGIN
                        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'feedback_liked_check') THEN
                            ALTER TABLE feedback ADD CONSTRAINT feedback_liked_check CHECK (liked IN (0, 1));
                        END IF;
                        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'feedback_meeting_agree_check') THEN
                            ALTER TABLE feedback ADD CONSTRAINT feedback_meeting_agree_check CHECK (meeting_agree IN (0, 1));
                        END IF;
                        IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'feedback_user_score_check') THEN
                            ALTER TABLE feedback ADD CONSTRAINT feedback_user_score_check
                                CHECK (user_score IS NULL OR (user_score >= 1 AND user_score <= 5));
                        END IF;
                    END$$;
                    """
                )

    # Сериализует данные в JSON для хранения.
    @staticmethod
    def _json(value: Dict[str, Any]) -> str:
        return json.dumps(value, ensure_ascii=False)

    # Работает с фотографиями анкеты.
    @staticmethod
    def _photo_from_row(row: Dict[str, Any]) -> PhotoRecord:
        return {
            "photo_id": int(row["photo_id"]),
            "photo_data": bytes(row["photo_data"]),
            "mime_type": str(row["mime_type"]),
            "filename": str(row["filename"]),
        }

    # Регистрирует пользователя в базе при первом обращении.
    def register_user(self, user_id: int) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users(user_id) VALUES (%s) ON CONFLICT(user_id) DO NOTHING",
                    (user_id,),
                )
                conn.commit()
                return cur.rowcount > 0

    # Работает с вопросами анкеты или теста.
    def save_questionnaire(self, user_id: int, answers: Dict[str, str]) -> None:
        self.register_user(user_id)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO questionnaire_answers(user_id, answers_json)
                    VALUES (%s, %s::jsonb)
                    ON CONFLICT(user_id) DO UPDATE SET
                        answers_json = EXCLUDED.answers_json,
                        updated_at = NOW()
                    """,
                    (user_id, self._json(answers)),
                )
                conn.commit()

    # Работает с анкетой или профилем пользователя.
    def save_text_profile(self, user_id: int, about_text: str) -> None:
        self.register_user(user_id)
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO text_profiles(user_id, about_text)
                    VALUES (%s, %s)
                    ON CONFLICT(user_id) DO UPDATE SET
                        about_text = EXCLUDED.about_text,
                        updated_at = NOW()
                    """,
                    (user_id, about_text),
                )
                conn.commit()

    # Работает с анкетой или профилем пользователя.
    def get_user_profile(self, user_id: int) -> Optional[UserProfile]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT answers_json FROM questionnaire_answers WHERE user_id = %s", (user_id,))
                q = cur.fetchone()
                cur.execute("SELECT about_text FROM text_profiles WHERE user_id = %s", (user_id,))
                t = cur.fetchone()
                if not q and not t:
                    return None
                cur.execute(
                    """
                    SELECT photo_id, photo_data, mime_type, filename
                    FROM user_photos
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                )
                photo_rows = cur.fetchall()
        return UserProfile(
            user_id=user_id,
            questionnaire=q["answers_json"] if q else {},
            about_text=t["about_text"] if t else "",
            photos=[self._photo_from_row(row) for row in photo_rows],
        )

    # Работает с анкетой или профилем пользователя.
    def get_all_profiles(self, exclude_user_id: Optional[int] = None) -> List[UserProfile]:
        sql = """
            SELECT u.user_id, qa.answers_json, tp.about_text
            FROM users u
            LEFT JOIN questionnaire_answers qa ON qa.user_id = u.user_id
            LEFT JOIN text_profiles tp ON tp.user_id = u.user_id
        """
        params: tuple[Any, ...] = ()
        if exclude_user_id is not None:
            sql += " WHERE u.user_id != %s"
            params = (exclude_user_id,)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()

                profiles = [
                    UserProfile(
                        user_id=int(row["user_id"]),
                        questionnaire=row["answers_json"] or {},
                        about_text=row["about_text"] or "",
                        photos=[],
                    )
                    for row in rows
                ]
                user_ids = [profile.user_id for profile in profiles]
                if user_ids:
                    cur.execute(
                        """
                        SELECT photo_id, user_id, photo_data, mime_type, filename
                        FROM user_photos
                        WHERE user_id = ANY(%s)
                        ORDER BY user_id, created_at DESC
                        """,
                        (user_ids,),
                    )
                    photo_rows = cur.fetchall()
                else:
                    photo_rows = []

        photos_by_user: Dict[int, List[PhotoRecord]] = {}
        for row in photo_rows:
            photos_by_user.setdefault(int(row["user_id"]), []).append(self._photo_from_row(row))
        for profile in profiles:
            profile.photos = photos_by_user.get(profile.user_id, [])
        return profiles

    # Работает с фотографиями анкеты.
    def add_user_photos(self, user_id: int, photos: List[Dict[str, Any]]) -> int:
        if not photos:
            return 0
        self.register_user(user_id)
        with self._connect() as conn:
            with conn.cursor() as cur:
                for photo in photos:
                    photo_data = photo.get("photo_data")
                    if not isinstance(photo_data, (bytes, bytearray, memoryview)) or not photo_data:
                        continue
                    cur.execute(
                        """
                        INSERT INTO user_photos(user_id, photo_data, mime_type, filename)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (
                            user_id,
                            bytes(photo_data),
                            photo.get("mime_type", "image/jpeg"),
                            photo.get("filename", "photo.jpg"),
                        ),
                    )
                cur.execute("SELECT COUNT(*) AS c FROM user_photos WHERE user_id = %s", (user_id,))
                total = cur.fetchone()
                conn.commit()
        return int(total["c"]) if total else 0

    # Работает с фотографиями анкеты.
    def get_user_photos(self, user_id: int) -> List[PhotoRecord]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT photo_id, photo_data, mime_type, filename
                    FROM user_photos
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    """,
                    (user_id,),
                )
                rows = cur.fetchall()
        return [self._photo_from_row(row) for row in rows]

    # Работает с анкетой или профилем пользователя.
    def clear_user_profile(self, user_id: int) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                for table in ("questionnaire_answers", "text_profiles", "user_photos", "user_drafts", "psychology_profile"):
                    cur.execute(f"DELETE FROM {table} WHERE user_id = %s", (user_id,))
                for table in ("likes", "dislikes", "blocks"):
                    cur.execute(f"DELETE FROM {table} WHERE from_user_id = %s OR to_user_id = %s", (user_id, user_id))
                conn.commit()

    # Сохраняет данные в базе.
    def save_like(self, from_user_id: int, to_user_id: int) -> None:
        self._execute(
            "INSERT INTO likes(from_user_id, to_user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (from_user_id, to_user_id),
        )

    # Проверяет наличие записи или состояния.
    def has_like(self, from_user_id: int, to_user_id: int) -> bool:
        return self._fetch_one(
            "SELECT 1 FROM likes WHERE from_user_id = %s AND to_user_id = %s",
            (from_user_id, to_user_id),
        ) is not None

    # Сохраняет данные в базе.
    def save_dislike(self, from_user_id: int, to_user_id: int) -> None:
        self._execute(
            "INSERT INTO dislikes(from_user_id, to_user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (from_user_id, to_user_id),
        )

    # Возвращает данные из хранилища или справочника.
    def get_viewed_user_ids(self, from_user_id: int) -> List[int]:
        rows = self._fetch_all(
            """
            SELECT to_user_id FROM likes WHERE from_user_id = %s
            UNION
            SELECT to_user_id FROM dislikes WHERE from_user_id = %s
            """,
            (from_user_id, from_user_id),
        )
        return [int(r["to_user_id"]) for r in rows]

    # Сохраняет данные в базе.
    def save_block(self, from_user_id: int, to_user_id: int) -> None:
        self._execute(
            "INSERT INTO blocks(from_user_id, to_user_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            (from_user_id, to_user_id),
        )

    # Сохраняет данные в базе.
    def save_report(self, from_user_id: int, to_user_id: int, reason: str) -> None:
        self._execute(
            "INSERT INTO reports(from_user_id, to_user_id, reason) VALUES (%s, %s, %s)",
            (from_user_id, to_user_id, reason),
        )

    # Возвращает данные из хранилища или справочника.
    def get_blocked_user_ids(self, from_user_id: int) -> List[int]:
        rows = self._fetch_all("SELECT to_user_id FROM blocks WHERE from_user_id = %s", (from_user_id,))
        return [int(r["to_user_id"]) for r in rows]

    # Возвращает данные из хранилища или справочника.
    def get_blocking_user_ids(self, to_user_id: int) -> List[int]:
        rows = self._fetch_all("SELECT from_user_id FROM blocks WHERE to_user_id = %s", (to_user_id,))
        return [int(r["from_user_id"]) for r in rows]

    # Сохраняет данные в базе.
    def save_draft(self, user_id: int, draft: Dict[str, Any]) -> None:
        self.register_user(user_id)
        self._execute(
            """
            INSERT INTO user_drafts(user_id, draft_json)
            VALUES (%s, %s::jsonb)
            ON CONFLICT(user_id) DO UPDATE SET
                draft_json = EXCLUDED.draft_json,
                updated_at = NOW()
            """,
            (user_id, self._json(draft)),
        )

    # Возвращает данные из хранилища или справочника.
    def get_draft(self, user_id: int) -> Optional[Dict[str, Any]]:
        row = self._fetch_one("SELECT draft_json FROM user_drafts WHERE user_id = %s", (user_id,))
        return row["draft_json"] if row else None

    # Удаляет сохраненные данные пользователя.
    def clear_draft(self, user_id: int) -> None:
        self._execute("DELETE FROM user_drafts WHERE user_id = %s", (user_id,))

    # Записывает событие или диагностическую информацию.
    def log_event(self, user_id: int, event_name: str, meta: Optional[Dict[str, Any]] = None) -> None:
        self.register_user(user_id)
        self._execute(
            "INSERT INTO events(user_id, event_name, meta_json) VALUES (%s, %s, %s::jsonb)",
            (user_id, event_name, self._json(meta or {})),
        )

    # Считает количество записей по условию.
    def count_events_today(self, user_id: int, event_name: str) -> int:
        row = self._fetch_one(
            """
            SELECT COUNT(*) AS c
            FROM events
            WHERE user_id = %s
              AND event_name = %s
              AND created_at::date = NOW()::date
            """,
            (user_id, event_name),
        )
        return int(row["c"]) if row else 0

    # Возвращает данные из хранилища или справочника.
    def get_reports(self, limit: int = 20) -> List[Dict[str, Any]]:
        return self._fetch_all("SELECT * FROM reports ORDER BY created_at DESC LIMIT %s", (limit,))

    # Возвращает данные из хранилища или справочника.
    def get_funnel_counts(self) -> Dict[str, int]:
        stages = ["start", "profile_complete", "browse_started", "like_sent", "match"]
        result: Dict[str, int] = {}
        with self._connect() as conn:
            with conn.cursor() as cur:
                for stage in stages:
                    cur.execute("SELECT COUNT(DISTINCT user_id) AS c FROM events WHERE event_name = %s", (stage,))
                    row = cur.fetchone()
                    result[stage] = int(row["c"]) if row else 0
        return result

    # Возвращает данные из хранилища или справочника.
    def get_monthly_admin_report(self, days: int = 30) -> Dict[str, Any]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) AS c FROM users WHERE registered_at >= NOW() - (%s || ' days')::interval",
                    (days,),
                )
                new_users = int(cur.fetchone()["c"])

                cur.execute(
                    """
                    SELECT COUNT(DISTINCT user_id) AS c
                    FROM events
                    WHERE created_at >= NOW() - (%s || ' days')::interval
                    """,
                    (days,),
                )
                active_users = int(cur.fetchone()["c"])

                cur.execute(
                    "SELECT COUNT(*) AS c FROM likes WHERE created_at >= NOW() - (%s || ' days')::interval",
                    (days,),
                )
                likes = int(cur.fetchone()["c"])

                cur.execute(
                    """
                    SELECT COUNT(*) AS c
                    FROM likes l1
                    JOIN likes l2
                      ON l1.from_user_id = l2.to_user_id
                     AND l1.to_user_id = l2.from_user_id
                    WHERE l1.created_at >= NOW() - (%s || ' days')::interval
                      AND l1.from_user_id < l1.to_user_id
                    """,
                    (days,),
                )
                matches = int(cur.fetchone()["c"])

                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS feedback_count,
                        COALESCE(AVG(user_score), 0) AS avg_score,
                        COALESCE(SUM(CASE WHEN liked = 1 AND meeting_agree = 1 THEN 1 ELSE 0 END), 0) AS successful
                    FROM feedback
                    WHERE created_at >= NOW() - (%s || ' days')::interval
                    """,
                    (days,),
                )
                feedback = cur.fetchone()

                cur.execute(
                    "SELECT COUNT(*) AS c FROM reports WHERE created_at >= NOW() - (%s || ' days')::interval",
                    (days,),
                )
                reports = int(cur.fetchone()["c"])

                cur.execute(
                    """
                    SELECT event_name, COUNT(*) AS c
                    FROM events
                    WHERE created_at >= NOW() - (%s || ' days')::interval
                    GROUP BY event_name
                    ORDER BY c DESC
                    LIMIT 10
                    """,
                    (days,),
                )
                top_events = cur.fetchall()

        return {
            "days": days,
            "new_users": new_users,
            "active_users": active_users,
            "likes": likes,
            "matches": matches,
            "feedback_count": int(feedback["feedback_count"]),
            "avg_score": round(float(feedback["avg_score"] or 0), 2),
            "successful_feedback": int(feedback["successful"]),
            "reports": reports,
            "top_events": top_events,
        }

    # Работает с фотографиями анкеты.
    def get_photo_storage_stats(self) -> Dict[str, int]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        COUNT(*) AS photos_count,
                        COUNT(DISTINCT user_id) AS users_with_photos,
                        COALESCE(SUM(octet_length(photo_data)), 0) AS total_bytes
                    FROM user_photos
                    """
                )
                row = cur.fetchone()
        return {
            "photos_count": int(row["photos_count"]),
            "users_with_photos": int(row["users_with_photos"]),
            "total_bytes": int(row["total_bytes"]),
        }

    # Возвращает данные из хранилища или справочника.
    def get_incoming_like_user_ids(self, user_id: int) -> List[int]:
        rows = self._fetch_all(
            """
            SELECT l.from_user_id
            FROM likes l
            WHERE l.to_user_id = %s
              AND NOT EXISTS (
                  SELECT 1 FROM likes l2
                  WHERE l2.from_user_id = %s AND l2.to_user_id = l.from_user_id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM dislikes d
                  WHERE d.from_user_id = %s AND d.to_user_id = l.from_user_id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM blocks b
                  WHERE b.from_user_id = %s AND b.to_user_id = l.from_user_id
              )
            ORDER BY l.created_at DESC
            """,
            (user_id, user_id, user_id, user_id),
        )
        return [int(r["from_user_id"]) for r in rows]

    # Работает с подбором и совместимостью анкет.
    def get_mutual_match_user_ids(self, user_id: int) -> List[int]:
        rows = self._fetch_all(
            """
            SELECT l1.to_user_id AS other_id
            FROM likes l1
            JOIN likes l2
              ON l1.from_user_id = l2.to_user_id
             AND l1.to_user_id = l2.from_user_id
            WHERE l1.from_user_id = %s
            ORDER BY l1.created_at DESC
            """,
            (user_id,),
        )
        return [int(r["other_id"]) for r in rows]

    # Работает с отзывами пользователей после встреч.
    def save_feedback(
        self, from_user_id: int, to_user_id: int, liked: int, meeting_agree: int, user_score: Optional[int]
    ) -> None:
        self._execute(
            """
            INSERT INTO feedback(from_user_id, to_user_id, liked, meeting_agree, user_score)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (from_user_id, to_user_id) DO NOTHING
            """,
            (from_user_id, to_user_id, liked, meeting_agree, user_score),
        )

    # Работает с отзывами пользователей после встреч.
    def has_feedback(self, from_user_id: int, to_user_id: int) -> bool:
        return self._fetch_one(
            "SELECT 1 FROM feedback WHERE from_user_id = %s AND to_user_id = %s LIMIT 1",
            (from_user_id, to_user_id),
        ) is not None

    # Работает с отзывами пользователей после встреч.
    def get_feedback_rows(self) -> List[Dict[str, Any]]:
        return self._fetch_all("SELECT * FROM feedback ORDER BY id")

    # Работает с отзывами пользователей после встреч.
    def get_feedback_rows_by_author(self, from_user_id: int) -> List[Dict[str, Any]]:
        return self._fetch_all("SELECT * FROM feedback WHERE from_user_id = %s ORDER BY id", (from_user_id,))

    # Работает с отзывами пользователей после встреч.
    def get_nlp_feedback_dataset_rows(self) -> List[Dict[str, Any]]:
        return self._fetch_all(
            """
            SELECT
                f.id,
                f.from_user_id,
                f.to_user_id,
                f.liked,
                f.meeting_agree,
                f.user_score,
                f.created_at,
                qa_from.answers_json AS from_answers,
                qa_to.answers_json AS to_answers,
                tp_from.about_text AS from_about,
                tp_to.about_text AS to_about
            FROM feedback f
            JOIN questionnaire_answers qa_from ON qa_from.user_id = f.from_user_id
            JOIN questionnaire_answers qa_to ON qa_to.user_id = f.to_user_id
            JOIN text_profiles tp_from ON tp_from.user_id = f.from_user_id
            JOIN text_profiles tp_to ON tp_to.user_id = f.to_user_id
            ORDER BY f.id
            """
        )

    # Сохраняет данные в базе.
    def save_psychology_answers(
        self, user_id: int, answers: Dict[str, int], scores: Optional[Dict[str, float]] = None
    ) -> None:
        self.register_user(user_id)
        scores_json = self._json(scores) if scores else None
        self._execute(
            """
            INSERT INTO psychology_profile(user_id, answers_json, scores_json)
            VALUES (%s, %s::jsonb, %s::jsonb)
            ON CONFLICT(user_id) DO UPDATE SET
                answers_json = EXCLUDED.answers_json,
                scores_json = EXCLUDED.scores_json,
                updated_at = NOW()
            """,
            (user_id, self._json(answers), scores_json),
        )

    # Возвращает данные из хранилища или справочника.
    def get_psychology_scores(self, user_id: int) -> Optional[Dict[str, float]]:
        row = self._fetch_one("SELECT scores_json FROM psychology_profile WHERE user_id = %s", (user_id,))
        return row["scores_json"] if row and row["scores_json"] else None

    # Работает с анкетой или профилем пользователя.
    def has_psychology_profile(self, user_id: int) -> bool:
        return self._fetch_one("SELECT 1 FROM psychology_profile WHERE user_id = %s", (user_id,)) is not None
