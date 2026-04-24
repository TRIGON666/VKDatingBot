#!/usr/bin/env python3
"""
Скрипт для мониторинга качества NLP модели и прогресса сбора данных.

Запуск:
    python monitor_nlp.py          # Общий отчет
    python monitor_nlp.py --last-hours 24  # Последние 24 часа
    python monitor_nlp.py --export report.json  # Сохранить отчет в JSON
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from src.config import NLP_DATA_PATH, NLP_MIN_EXAMPLES
from src.nlp_data_collector import get_nlp_stats
from src.nlp_metrics import NLPMetricsTracker


def format_percentage(value: float) -> str:
    """Преобразовать число в строку процента."""
    if isinstance(value, (int, float)):
        return f"{value:.1%}"
    return str(value)


def print_separator(title: str = "") -> None:
    """Вывести разделитель."""
    if title:
        print(f"\n{'=' * 60}\n{title}\n{'=' * 60}")
    else:
        print("=" * 60)


def safe_metrics(tracker: NLPMetricsTracker, hours: int | None) -> dict:
    """Безопасно получить метрики без падения скрипта."""
    try:
        return tracker.calculate_metrics(hours=hours)
    except Exception:
        return {
            "predictions_count": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "period": "all" if hours is None else f"{hours}h",
        }


def main() -> int:
    parser = argparse.ArgumentParser(description="Мониторинг качества NLP и прогресса сбора данных")
    parser.add_argument(
        "--last-hours",
        type=int,
        default=None,
        help="Показать метрики за последние N часов (по умолчанию: все время)",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Сохранить отчет в JSON",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=NLP_DATA_PATH,
        help="Путь к CSV с данными",
    )

    args = parser.parse_args()
    tracker = NLPMetricsTracker()
    stats = get_nlp_stats(args.data_path)

    print_separator("ОТЧЕТ МОНИТОРИНГА NLP")
    print(f"Сформирован: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print_separator("ПРОГРЕСС СБОРА ДАННЫХ")

    total = int(stats.get("total", 0))
    positive = int(stats.get("positive", 0))
    neutral = int(stats.get("neutral", 0))
    negative = int(stats.get("negative", 0))
    print(f"Всего примеров: {total}")
    if total > 0:
        print(f"  Положительных: {positive:4d} ({positive / total:5.1%})")
        print(f"  Нейтральных:   {neutral:4d} ({neutral / total:5.1%})")
        print(f"  Отрицательных: {negative:4d} ({negative / total:5.1%})")
    else:
        print("Пока нет данных. Бот создаст CSV после первых взаимодействий.")

    min_required = NLP_MIN_EXAMPLES
    if total >= min_required:
        print(f"\nГотово к обучению: {total} >= {min_required}")
        print("Команда: python train_nlp_model.py")
    else:
        remaining = min_required - total
        print(f"\nДо обучения нужно еще примеров: {remaining}")

    print_separator("МЕТРИКИ КАЧЕСТВА")

    metrics = safe_metrics(tracker, args.last_hours if args.last_hours else None)
    print(f"Период: {metrics.get('period', 'all')}")
    print(f"  Точность (Accuracy): {format_percentage(float(metrics.get('accuracy', 0.0)))}")
    print(f"  Точность по положительному классу (Precision): {format_percentage(float(metrics.get('precision', 0.0)))}")
    print(f"  Полнота (Recall): {format_percentage(float(metrics.get('recall', 0.0)))}")
    print(f"  F1-мера: {format_percentage(float(metrics.get('f1', 0.0)))}")
    print(f"  Предсказаний учтено: {int(metrics.get('predictions_count', 0))}")

    print_separator("ДИНАМИКА")

    metrics_24h = safe_metrics(tracker, 24)
    metrics_7d = safe_metrics(tracker, 24 * 7)
    metrics_all = safe_metrics(tracker, None)

    accuracy_24h = float(metrics_24h.get("accuracy", 0.0))
    accuracy_7d = float(metrics_7d.get("accuracy", 0.0))
    accuracy_all = float(metrics_all.get("accuracy", 0.0))

    print("Точность (Accuracy):")
    print(f"  За 24ч:   {format_percentage(accuracy_24h)}")
    print(f"  За 7д:    {format_percentage(accuracy_7d)}")
    print(f"  Все время:{format_percentage(accuracy_all)}")

    if int(metrics_24h.get("predictions_count", 0)) > 0:
        if accuracy_24h > accuracy_7d:
            print("Тренд: улучшается")
        elif accuracy_24h < accuracy_7d:
            print("Тренд: ухудшается")
        else:
            print("Тренд: стабильный")
    else:
        print("Тренд: недостаточно данных")

    print_separator("РЕКОМЕНДАЦИИ")

    report = {}
    if total >= NLP_MIN_EXAMPLES:
        print("Данных достаточно для обучения. Запустите: python train_nlp_model.py")
        report["action"] = "TRAIN"
    else:
        print(f"Продолжайте сбор данных: {total}/{NLP_MIN_EXAMPLES}")
        report["action"] = "COLLECT_MORE"

    if accuracy_all >= 0.8:
        print("Качество модели хорошее (accuracy > 80%).")
    elif accuracy_all >= 0.7:
        print("Качество модели приемлемое (70-80%).")
    else:
        print("Качество модели низкое (<70%), рекомендуется дообучение.")

    report["examples_collected"] = total
    report["examples_needed"] = max(0, NLP_MIN_EXAMPLES - total)
    report["accuracy_all"] = accuracy_all

    print_separator()
    print()

    if args.export:
        report.update(
            {
            "timestamp": datetime.now().isoformat(),
            "data_path": args.data_path,
            "stats": stats,
            "metrics_selected": metrics,
            "metrics_24h": metrics_24h,
            "metrics_7d": metrics_7d,
            "metrics_all": metrics_all,
            }
        )

        output_dir = os.path.dirname(args.export)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(args.export, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        print(f"Отчет сохранен: {args.export}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
