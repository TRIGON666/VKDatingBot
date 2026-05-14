from __future__ import annotations

import argparse
import importlib
import json
import os
import sys

import psycopg

from src.config import NLP_MODEL_PATH, NLP_REPORT_PATH, get_settings
from src.database import Database
from src.logging_config import setup_logging
from src.vk_bot import VKCompatibilityBot
def _run_bot() -> int:
    settings = get_settings()
    setup_logging(settings.log_file, settings.log_level)
    if not settings.vk_token:
        raise RuntimeError("VK_TOKEN is not set. Copy .env.example to .env and paste your VK community token.")

    try:
        db = Database(settings.database_url)
    except psycopg.OperationalError as exc:
        raise RuntimeError(
            "Could not connect to PostgreSQL. Start the project database with "
            "`docker compose up -d postgres` and check DATABASE_URL in .env. "
            "By default the project uses localhost:5433 to avoid conflicts with a local PostgreSQL."
        ) from exc
    bot = VKCompatibilityBot(
        token=settings.vk_token,
        db=db,
        admin_ids=settings.admin_ids,
        group_id=settings.vk_group_id or None,
    )
    print("Bot is running. Press Ctrl+C to stop.")
    bot.run()
    return 0
def _check_env() -> int:
    settings = get_settings()
    setup_logging(settings.log_file, settings.log_level)

    checks: list[tuple[str, bool, str]] = []
    checks.append(("VK_TOKEN", bool(settings.vk_token), "set" if settings.vk_token else "missing"))
    checks.append(("VK_GROUP_ID", True, str(settings.vk_group_id) if settings.vk_group_id else "not set; bot will try to detect it"))
    checks.append(("DATABASE_URL", bool(settings.database_url), settings.database_url))
    admin_message = ",".join(str(v) for v in sorted(settings.admin_ids)) or "not set; admin commands are disabled"
    checks.append(("ADMIN_IDS", True, admin_message))
    checks.append(("LOG_FILE", bool(settings.log_file), settings.log_file))
    checks.append(("NLP model", os.path.exists(NLP_MODEL_PATH), NLP_MODEL_PATH))
    checks.append(("NLP train dataset", os.path.exists("exports/nlp_dataset_train.csv"), "exports/nlp_dataset_train.csv"))

    db_ok = False
    db_message = ""
    try:
        db = Database(settings.database_url)
        db.get_funnel_counts()
        photo_stats = db.get_photo_storage_stats()
        db_ok = True
        db_message = (
            "connected; "
            f"photos={photo_stats.get('photos_count', 0)}, "
            f"users_with_photos={photo_stats.get('users_with_photos', 0)}"
        )
    except Exception as exc:
        db_message = str(exc).splitlines()[0][:180]
    checks.append(("PostgreSQL", db_ok, db_message))

    print("Environment check:")
    for name, ok, message in checks:
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {name}: {message}")
    return 0 if all(ok for _, ok, _ in checks) else 1
def _run_legacy_module(module_name: str, extra_args: list[str]) -> int:
    module = importlib.import_module(module_name)
    module_main = getattr(module, "main", None)
    if not callable(module_main):
        raise RuntimeError(f"Module '{module_name}' does not expose callable main()")

    prev_argv = sys.argv
    sys.argv = [f"{module_name}.py", *extra_args]
    try:
        result = module_main()
    finally:
        sys.argv = prev_argv

    return int(result) if isinstance(result, int) else 0
def _run_tools(extra_args: list[str]) -> int:
    tools_parser = argparse.ArgumentParser(prog="main.py tools", description="NLP helper tools")
    subparsers = tools_parser.add_subparsers(dest="tool_cmd", required=True)

    subparsers.add_parser("export-dataset", help="Export NLP dataset from the local database")
    subparsers.add_parser("train-text-model", help="Train text compatibility model")

    check_parser = subparsers.add_parser("check-compatibility", help="Predict compatibility for two texts")
    check_parser.add_argument("--left", required=True, help="Left profile text")
    check_parser.add_argument("--right", required=True, help="Right profile text")
    check_parser.add_argument("--model", default=NLP_MODEL_PATH, help="Path to trained model")

    args = tools_parser.parse_args(extra_args)

    if args.tool_cmd == "export-dataset":
        from src.analytics import export_nlp_dataset_csv

        db = Database(get_settings().database_url)
        output_path = export_nlp_dataset_csv(db, "exports/nlp_dataset.csv")
        print(f"NLP dataset exported: {output_path}")
        return 0

    if args.tool_cmd == "train-text-model":
        from src.nlp_compatibility import train_text_compatibility_model

        report = train_text_compatibility_model(
            dataset_csv_path="exports/nlp_dataset_train.csv",
            model_output_path=NLP_MODEL_PATH,
            report_output_path=NLP_REPORT_PATH,
            validation_csv_path="exports/nlp_dataset_val.csv",
            test_csv_path="exports/nlp_dataset_test.csv",
            extra_dataset_paths=["data/nlp_training_data.csv"],
            real_data_weight=4,
            random_seed=42,
        )
        print("Text compatibility model trained.")
        print(f"Model:  {NLP_MODEL_PATH}")
        print(f"Report: {NLP_REPORT_PATH}")
        print(f"Accuracy: {report['accuracy']}")
        print(f"Macro F1: {report['macro_f1']}")
        if "final_test_accuracy" in report:
            print(f"Final test accuracy: {report['final_test_accuracy']}")
            print(f"Final test macro F1: {report['final_test_macro_f1']}")
        return 0

    from src.nlp_compatibility import predict_text_compatibility

    result = predict_text_compatibility(args.left, args.right, args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VK bot and project utilities")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["bot", "check-env", "monitor", "train", "analyze", "auto-train", "utils", "tools", "nlp"],
        default="bot",
        help="Command to run. Default: bot",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to selected command")
    return parser


# Точка входа: разбирает аргументы и запускает нужный сценарий.
def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    ns = parser.parse_args(argv)

    if ns.command == "bot":
        return _run_bot()

    if ns.command == "check-env":
        return _check_env()

    if ns.command in {"tools", "nlp"}:
        return _run_tools(list(ns.args or []))

    module_by_command = {
        "monitor": "monitor_nlp",
        "train": "train_nlp_model",
        "analyze": "analyze_nlp",
        "auto-train": "auto_train_nlp",
        "utils": "nlp_utils",
    }
    forwarded_args = list(ns.args or [])
    if forwarded_args and forwarded_args[0] == "--":
        forwarded_args = forwarded_args[1:]
    return _run_legacy_module(module_by_command[ns.command], forwarded_args)


if __name__ == "__main__":
    raise SystemExit(main())
