from __future__ import annotations

import argparse
import importlib
import json
import sys

from src.config import get_settings
from src.database import Database
from src.vk_bot import VKCompatibilityBot


def _run_bot() -> int:
    settings = get_settings()
    if not settings.vk_token:
        raise RuntimeError("Не задан VK_TOKEN. Скопируйте .env.example в .env и укажите токен.")
    if not settings.database_url:
        raise RuntimeError("Не задан DATABASE_URL. Укажите DSN PostgreSQL в .env.")

    db = Database(settings.database_url)
    bot = VKCompatibilityBot(token=settings.vk_token, db=db)
    bot.run()
    return 0


def _run_legacy_module(module_name: str, extra_args: list[str]) -> int:
    module = importlib.import_module(module_name)
    module_main = getattr(module, "main", None)
    if not callable(module_main):
        raise RuntimeError(f"Модуль '{module_name}' не содержит вызываемую функцию main()")

    prev_argv = sys.argv
    sys.argv = [f"{module_name}.py", *extra_args]
    try:
        result = module_main()
    finally:
        sys.argv = prev_argv

    return int(result) if isinstance(result, int) else 0


def _run_tools(extra_args: list[str]) -> int:
    tools_parser = argparse.ArgumentParser(prog="main.py tools", description="Вспомогательные NLP-инструменты")
    subparsers = tools_parser.add_subparsers(dest="tool_cmd", required=True)

    export_parser = subparsers.add_parser("export-dataset", help="Экспортировать NLP-датасет из БД")
    export_parser.set_defaults(tool_cmd="export-dataset")

    train_parser = subparsers.add_parser("train-text-model", help="Обучить модель текстовой совместимости")
    train_parser.set_defaults(tool_cmd="train-text-model")

    check_parser = subparsers.add_parser("check-compatibility", help="Предсказать совместимость двух текстов")
    check_parser.add_argument("--left", required=True, help="Текст левого профиля")
    check_parser.add_argument("--right", required=True, help="Текст правого профиля")
    check_parser.add_argument("--model", default="models/nlp_model.pkl", help="Путь к обученной модели")
    check_parser.set_defaults(tool_cmd="check-compatibility")

    args = tools_parser.parse_args(extra_args)

    if args.tool_cmd == "export-dataset":
        from src.analytics import export_nlp_dataset_csv

        settings = get_settings()
        if not settings.database_url:
            raise RuntimeError("Не задан DATABASE_URL. Укажите DSN PostgreSQL в .env.")
        db = Database(settings.database_url)
        output_path = export_nlp_dataset_csv(db, "exports/nlp_dataset.csv")
        print(f"NLP-датасет экспортирован: {output_path}")
        return 0

    if args.tool_cmd == "train-text-model":
        from src.config import NLP_MODEL_PATH, NLP_REPORT_PATH
        from src.nlp_compatibility import train_text_compatibility_model

        report = train_text_compatibility_model(
            dataset_csv_path="exports/nlp_dataset_train.csv",
            model_output_path=NLP_MODEL_PATH,
            report_output_path=NLP_REPORT_PATH,
            validation_csv_path="exports/nlp_dataset_val.csv",
            test_csv_path="exports/nlp_dataset_test.csv",
            random_seed=42,
        )
        print("Модель текстовой совместимости обучена.")
        print(f"Модель:   {NLP_MODEL_PATH}")
        print(f"Отчет:    {NLP_REPORT_PATH}")
        print(f"Точность (Accuracy): {report['accuracy']}")
        print(f"Macro F1: {report['macro_f1']}")
        if "final_test_accuracy" in report:
            print(f"Точность на финальном тесте: {report['final_test_accuracy']}")
            print(f"Macro F1 на финальном тесте: {report['final_test_macro_f1']}")
        return 0

    from src.nlp_compatibility import predict_text_compatibility

    result = predict_text_compatibility(args.left, args.right, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Единый запускатор VK-бота и NLP-инструментов",
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=["bot", "monitor", "train", "analyze", "auto-train", "utils", "tools"],
        default="bot",
        help="Что запустить (по умолчанию: bot)",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Аргументы, передаваемые выбранной команде",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)

    if ns.command == "bot":
        return _run_bot()

    if ns.command == "tools":
        return _run_tools(list(ns.args or []))

    module_by_command = {
        "monitor": "monitor_nlp",
        "train": "train_nlp_model",
        "analyze": "analyze_nlp",
        "auto-train": "auto_train_nlp",
        "utils": "nlp_utils",
    }
    module_name = module_by_command[ns.command]
    forwarded_args = list(ns.args or [])
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    return _run_legacy_module(module_name, forwarded_args)


if __name__ == "__main__":
    raise SystemExit(main())
