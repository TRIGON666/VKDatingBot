#!/usr/bin/env python3
"""
Скрипт для переобучения NLP модели когда накопится достаточно данных.
Используйте когда будет 500+ примеров в data/nlp_training_data.csv

Запуск:
    python train_nlp_model.py

Или с параметрами:
    python train_nlp_model.py --min-examples 300 --output-model models/nlp_model.pkl
"""

import argparse
import os
import sys

from src.config import NLP_DATA_PATH, NLP_MIN_EXAMPLES, NLP_MODEL_PATH, NLP_REPORT_PATH
from src.nlp_compatibility import ensure_parent_dir, extract_macro_metrics, train_text_compatibility_model
from src.nlp_data_collector import get_nlp_stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Переобучение NLP модели на собранных данных"
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=NLP_MIN_EXAMPLES,
        help=f"Минимум примеров для обучения (по умолчанию: {NLP_MIN_EXAMPLES})",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=NLP_DATA_PATH,
        help="Путь к CSV с данными",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default=NLP_MODEL_PATH,
        help="Путь сохранения модели",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=NLP_REPORT_PATH,
        help="Путь сохранения отчета",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Обучать даже если примеров меньше минимума",
    )

    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        print(f"Ошибка: файл данных не найден: {args.data_path}")
        print("Сначала запустите бота, чтобы собрать данные.")
        return 1

    print("Текущая статистика данных:")
    try:
        stats = get_nlp_stats(args.data_path)
        print(f"  Всего примеров: {stats['total']}")
        print(f"  Положительных (лайк): {stats['positive']}")
        print(f"  Нейтральных (пропуск): {stats['neutral']}")
        print(f"  Отрицательных (блок): {stats['negative']}")
        print()
    except Exception as e:
        print(f"Ошибка чтения статистики: {e}")
        return 1

    if stats["total"] < args.min_examples and not args.force:
        required = args.min_examples - stats["total"]
        print("Пока недостаточно примеров для качественного обучения.")
        print(f"Нужно еще: {required}")
        print(f"Текущее количество: {stats['total']} / {args.min_examples}")
        return 0

    ensure_parent_dir(args.output_model)
    ensure_parent_dir(args.output_report)

    print(f"Запуск обучения на {stats['total']} примерах...")
    print()

    try:
        report_payload = train_text_compatibility_model(
            dataset_csv_path=args.data_path,
            model_output_path=args.output_model,
            report_output_path=args.output_report,
        )

        precision, recall, f1 = extract_macro_metrics(report_payload)

        print("Обучение завершено успешно.")
        print()
        print("Метрики качества:")
        print(f"  Точность (Accuracy): {float(report_payload.get('accuracy', 0.0)):.2%}")
        print(f"  Точность по положительному классу (Precision): {precision:.2%}")
        print(f"  Полнота (Recall): {recall:.2%}")
        print(f"  F1-мера: {f1:.2%}")
        print()
        print(f"Модель сохранена: {args.output_model}")
        print(f"Отчет сохранен: {args.output_report}")
        print()

        cm = report_payload.get("confusion_matrix")
        if isinstance(cm, list) and cm:
            labels = ["отрицательный", "нейтральный", "положительный"]
            print("Матрица ошибок (строки = факт, столбцы = прогноз):")
            for i, row in enumerate(cm):
                row_name = labels[i] if i < len(labels) else f"класс_{i}"
                print(f"  {row_name:8s}: {row}")
        print()
        return 0

    except Exception as e:
        print(f"Ошибка во время обучения: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
