from __future__ import annotations

import logging
from pathlib import Path


# Настраивает вывод логов в консоль и файл.
def setup_logging(log_file: str = "logs/bot.log", level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    console.setLevel(numeric_level)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(numeric_level)

    root.addHandler(console)
    root.addHandler(file_handler)
