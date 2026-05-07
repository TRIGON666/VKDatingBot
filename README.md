# VK Compatibility Bot

VK-бот для анкет, мэтчей и расчета совместимости. Проект запускается локально на Python и PostgreSQL, поэтому его можно передать через GitHub: другой пользователь копирует `.env.example`, запускает Docker Compose и стартует `main.py`.

Проект совместим с Windows, macOS и Linux. Основная разница только в командах активации виртуального окружения и копирования `.env`.

## Стек

- Python 3.10+
- PostgreSQL 16
- Docker Compose
- `vk-api`
- `psycopg`
- `scikit-learn`
- `sentence-transformers`
- `pandas`, `numpy`, `joblib`

## Быстрый Запуск На Windows

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
docker compose up -d postgres
python main.py check-env
python main.py
```

## Быстрый Запуск На macOS/Linux

На macOS заранее установите Docker Desktop и Python. Если Python ставится через Homebrew:

```bash
brew install python git
```

Дальше запуск такой:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
docker compose up -d postgres
python main.py check-env
python main.py
```

Если команда `docker compose` не найдена, проверьте, что Docker Desktop запущен и установлен Compose V2.

При успешном старте в консоли появится:

```text
Bot is running. Press Ctrl+C to stop.
```

## `.env`

После копирования `.env.example` заполните `.env`:

```env
VK_TOKEN=your_vk_community_token
VK_GROUP_ID=123456789
DATABASE_URL=postgresql://compatibility_user:compatibility_pass@localhost:5433/compatibility_bot
ADMIN_IDS=123456789
LOG_FILE=logs/bot.log
LOG_LEVEL=INFO
NLP_PRETRAINED_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

`ADMIN_IDS` - это VK ID администраторов через запятую. Если оставить пустым, админ-команды будут отключены.

`VK_GROUP_ID` - числовой ID сообщества без минуса. Он нужен для Bot Long Poll, чтобы получать полные вложения фото с URL.

Проект использует PostgreSQL на порту `5433`, чтобы не конфликтовать с локальным PostgreSQL на стандартном `5432`.

## Проверка Окружения

```bash
python main.py check-env
```

Проверяется:

- наличие `VK_TOKEN`;
- строка подключения к PostgreSQL;
- подключение к базе;
- наличие NLP-модели;
- наличие обучающего датасета;
- статистика фото в базе;
- состояние `ADMIN_IDS`.

## Команды

Основной запуск:

```bash
python main.py
```

NLP-утилиты:

```bash
python main.py nlp export-dataset
python main.py nlp train-text-model
python main.py nlp check-compatibility --left "люблю путешествия" --right "обожаю походы"
```

Старые скрипты оставлены как совместимые команды через `main.py`:

```bash
python main.py monitor
python main.py train
python main.py analyze --errors
python main.py auto-train
python main.py utils --status
```

## Админ-Команды В VK

Доступны только пользователям из `ADMIN_IDS`:

- `/admin_reports` - последние жалобы.
- `/admin_funnel` - базовая воронка событий.
- `/admin_month` - отчет за последние 30 дней: пользователи, лайки, мэтчи, отзывы, жалобы, топ событий.
- `/admin_photo_check` - диагностика фото: количество фото в базе, размер, проверка выгрузки тестового фото через VK API.

## Фото

Фото сохраняются в PostgreSQL в таблице `user_photos`. При загрузке бот:

- принимает фото из вложений VK;
- скачивает лучший доступный размер;
- проверяет MIME-тип: `image/jpeg`, `image/png`, `image/webp`;
- отсекает файлы больше 10 МБ;
- сохраняет байты, MIME-тип и имя файла.

Для загрузки фото бот должен работать в режиме Bot Long Poll. В логах при старте должна быть строка `LongPoll mode: bot`; режим `legacy` может получать только ID вложения без URL и не сможет скачать фото.

При показе анкеты бот выгружает фото из базы в сообщения VK через `photos.getMessagesUploadServer` и `photos.saveMessagesPhoto`.

Проверка:

```bash
python main.py check-env
```

и в VK:

```text
/admin_photo_check
```

## Логирование

Логи пишутся в консоль и файл из `LOG_FILE`. По умолчанию:

```text
logs/bot.log
```

Путь `logs/bot.log` одинаково работает на Windows, macOS и Linux: директория создается автоматически.

## База Данных

Остановить PostgreSQL:

```bash
docker compose down
```

Остановить и удалить локальные данные:

```bash
docker compose down -v
```

Подключиться к базе:

```bash
docker compose exec postgres psql -U compatibility_user -d compatibility_bot
```

Схема создается автоматически при первом запуске `Database`.

## Совместимость И NLP

Описание алгоритмов, весов, TF-IDF, NLP-модели и схем для диплома находится в:

[docs/compatibility_algorithms.md](docs/compatibility_algorithms.md)

Коротко:

- анкета считает совпадения по структурированным ответам;
- TF-IDF сравнивает похожесть текстов по важным словам и фразам;
- NLP использует предобученную multilingual-модель для смысловой близости текстов;
- итоговое ранжирование смешивает анкету, психологический профиль, TF-IDF/NLP и поведение пользователей.

Модель можно дообучать:

```bash
python main.py nlp train-text-model
```

Для проверки интеграции:

```bash
python test_nlp_integration.py
```

## Передача Через GitHub

Минимальный сценарий для другого пользователя на macOS/Linux:

```bash
git clone <repo-url>
cd <repo-folder>
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
docker compose up -d postgres
python main.py check-env
python main.py
```

Минимальный сценарий для Windows:

```powershell
git clone <repo-url>
cd <repo-folder>
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
docker compose up -d postgres
python main.py check-env
python main.py
```

Перед запуском нужно вставить свой `VK_TOKEN`, `VK_GROUP_ID` и при необходимости `ADMIN_IDS` в `.env`.

## Примечания Для macOS

- На Apple Silicon зависимости `torch` и `sentence-transformers` могут ставиться дольше, это нормально.
- Если `pip install -r requirements.txt` падает на сборке пакетов, обновите pip: `python -m pip install --upgrade pip setuptools wheel`.
- Если порт `5433` занят, измените порт в `docker-compose.yml` и `DATABASE_URL`.
- Если терминал показывает русские тексты некорректно, проверьте кодировку файла: проект хранит `.py`, `.md`, `.csv` в UTF-8.
