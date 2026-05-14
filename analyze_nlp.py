#!/usr/bin/env python3
"""
Интерактивный анализ и оптимизация NLP модели.

Команды:
    python analyze_nlp.py --export results.json  # Анализ и сохранение результатов
    python analyze_nlp.py --errors               # Анализ ошибочных предсказаний
"""

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

from src.config import NLP_DATA_PATH, NLP_MIN_EXAMPLES, NLP_MODEL_PATH
from src.nlp_compatibility import predict_text_compatibility
from src.nlp_data_collector import get_nlp_stats
from src.nlp_metrics import NLPMetricsTracker


class NLPAnalyzer:
    # Инициализирует объект и сохраняет нужные зависимости.
    def __init__(self, model_path=NLP_MODEL_PATH, data_path=NLP_DATA_PATH):
        self.model_path = model_path
        self.data_path = data_path
        self.tracker = NLPMetricsTracker()

    # Загружает CSV-примеры для анализа качества NLP.
    def load_data(self):
        """Load training data"""
        if not os.path.exists(self.data_path):
            return []

        data = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "text_left": row.get("text_left", ""),
                    "text_right": row.get("text_right", ""),
                    "label": row.get("label", "unknown"),
                })
        return data

    # Показывает распределение классов в NLP-данных.
    def analyze_statistics(self):
        """Analyze data distribution and statistics"""
        print("\n" + "=" * 60)
        print("СТАТИСТИКА ДАННЫХ")
        print("=" * 60)

        stats = get_nlp_stats(self.data_path)
        print(f"\nВсего примеров: {stats['total']}")

        if stats["total"] == 0:
            print("Данные пока не собраны")
            return stats

        print("\nРаспределение классов:")
        categories = {
            "positive": stats["positive"],
            "neutral": stats["neutral"],
            "negative": stats["negative"],
        }

        for cat, count in categories.items():
            pct = count / stats["total"]
            bar = "█" * int(pct * 40)
            print(f"  {cat:10s}: {count:4d} ({pct:5.1%}) {bar}")

        print("\nБаланс классов:")
        max_cat = max(categories.values())
        min_cat = min(categories.values())
        imbalance = (max_cat - min_cat) / max_cat if max_cat > 0 else 0.0

        if imbalance > 0.5:
            print(f"  Сильный перекос классов ({imbalance:.0%})")
            print("  Рекомендация: собрать больше примеров недопредставленного класса")
        elif imbalance > 0.3:
            print(f"  Умеренный перекос классов ({imbalance:.0%})")
        else:
            print("  Баланс классов хороший")

        return stats

    # Показывает накопленные метрики качества модели.
    def analyze_model_performance(self):
        """Analyze current model performance"""
        print("\n" + "=" * 60)
        print("КАЧЕСТВО МОДЕЛИ")
        print("=" * 60)

        try:
            metrics = self.tracker.calculate_metrics(hours=None)

            if not metrics or metrics.get("predictions_count", 0) == 0:
                print("Метрики предсказаний пока не накоплены")
                return {}

            print("\nМетрики (за все время):")
            print(f"  Предсказаний: {metrics.get('predictions_count', 0)}")
            print(f"  Точность (Accuracy): {metrics.get('accuracy', 0):.1%}")
            print(f"  Точность по положительному классу (Precision): {metrics.get('precision', 0):.1%}")
            print(f"  Полнота (Recall): {metrics.get('recall', 0):.1%}")
            print(f"  F1-мера: {metrics.get('f1', 0):.1%}")

            accuracy = metrics.get("accuracy", 0)
            print("\nОценка качества:")

            if accuracy >= 0.85:
                print("  Отличное качество (>=85%)")
            elif accuracy >= 0.75:
                print("  Хорошее качество (75-85%)")
            elif accuracy >= 0.65:
                print("  Приемлемое качество (65-75%)")
                print("  Рекомендация: выполнить переобучение")
            else:
                print("  Низкое качество (<65%)")
                print("  Рекомендация: срочно переобучить модель")

            return metrics

        except Exception as e:
            print(f"Не удалось рассчитать метрики: {e}")
            return {}

    # Ищет примеры, где модель ошибается.
    def find_problem_areas(self, sample_size: int = 100):
        """Analyze where model makes mistakes"""
        print("\n" + "=" * 60)
        print("АНАЛИЗ ПРОБЛЕМНЫХ ПРИМЕРОВ")
        print("=" * 60)

        if not os.path.exists(self.model_path):
            print(f"Файл модели не найден: {self.model_path}")
            print("Сначала обучите модель: python train_nlp_model.py")
            return

        data = self.load_data()
        if not data:
            print("Нет данных для анализа")
            return

        print(f"\nАнализируем до {sample_size} примеров из {len(data)}...")

        errors = []
        correct = []

        for example in data[:sample_size]:
            try:
                text_left = example.get("text_left", "")
                text_right = example.get("text_right", "")
                true_label = example.get("label", "unknown")

                if not text_left or not text_right:
                    continue

                pred = predict_text_compatibility(
                    text_left,
                    text_right,
                    model_path=self.model_path,
                )
                pred_label = str(pred.get("label", "neutral"))
                confidence = float(pred.get("score_percent", 0.0)) / 100.0

                if pred_label == true_label:
                    correct.append((pred_label, confidence))
                else:
                    errors.append(
                        {
                            "text_left": (text_left[:50] + "...") if len(text_left) > 50 else text_left,
                            "text_right": (text_right[:50] + "...") if len(text_right) > 50 else text_right,
                            "predicted": pred_label,
                            "true": true_label,
                            "confidence": confidence,
                        }
                    )

            except Exception:
                pass

        total_checked = len(correct) + len(errors)
        if total_checked == 0:
            print("Нет валидных примеров для анализа")
            return

        if errors:
            print(f"\nОшибок: {len(errors)} из {total_checked}")
            print("\nПроблемные пары:")

            for i, err in enumerate(errors[:5], 1):
                print(
                    f"\n  {i}. Прогноз: {err['predicted']}, "
                    f"факт: {err['true']} (уверенность: {err['confidence']:.1%})"
                )
                print(f"     Левый профиль:  {err['text_left']}")
                print(f"     Правый профиль: {err['text_right']}")
        else:
            accuracy = (len(correct) / total_checked) * 100
            print(f"\nНа выбранной выборке ошибок не найдено, точность: {accuracy:.1f}%")

    # Печатает рекомендации по улучшению NLP-данных и модели.
    def get_recommendations(self, stats: dict, metrics: dict):
        """Generate improvement recommendations"""
        print("\n" + "=" * 60)
        print("РЕКОМЕНДАЦИИ")
        print("=" * 60)

        total = int(stats.get("total", 0))
        print("\nДействия:")

        if total == 0:
            print("  1. Запустите бота для сбора данных")
            return

        if total < 100:
            print(f"  1. Соберите больше данных (еще {100 - total} примеров)")
        elif total < NLP_MIN_EXAMPLES:
            print(f"  1. Продолжайте сбор ({NLP_MIN_EXAMPLES - total} до минимального порога обучения)")
        else:
            print("  1. Можно обучать модель")
            print("     Команда: python train_nlp_model.py")

        categories = [stats["positive"], stats["neutral"], stats["negative"]]
        if categories and max(categories) > 0 and (max(categories) - min(categories)) / max(categories) > 0.5:
            print("  2. Выравнивайте классы (сильный перекос)")

        if float(metrics.get("accuracy", 0.0)) < 0.7 and int(metrics.get("predictions_count", 0)) > 0:
            print("  3. Переобучите модель (низкая точность)")

    # Сохраняет отчет анализа в JSON.
    def export_report(self, filepath: str, stats: dict, metrics: dict):
        """Export analysis report to JSON"""
        report = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "data_path": self.data_path,
            "model_path": self.model_path,
            "stats": stats,
            "metrics": metrics,
        }

        output_dir = os.path.dirname(filepath)
        if output_dir:
            Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)

        print(f"\nОтчет сохранен: {filepath}")

# Точка входа: разбирает аргументы и запускает нужный сценарий.
def main():
    parser = argparse.ArgumentParser(description="Анализ качества NLP модели")
    parser.add_argument("--data-path", default=NLP_DATA_PATH)
    parser.add_argument("--model-path", default=NLP_MODEL_PATH)
    parser.add_argument("--export", help="Сохранить отчет в JSON")
    parser.add_argument("--errors", action="store_true", help="Показать ошибочные предсказания")
    parser.add_argument("--sample-size", type=int, default=100, help="Размер выборки для анализа ошибок")

    args = parser.parse_args()

    analyzer = NLPAnalyzer(
        model_path=args.model_path,
        data_path=args.data_path,
    )

    stats = analyzer.analyze_statistics()
    metrics = analyzer.analyze_model_performance()

    if args.errors:
        analyzer.find_problem_areas(sample_size=max(1, args.sample_size))

    analyzer.get_recommendations(stats, metrics)

    if args.export:
        analyzer.export_report(args.export, stats, metrics)


if __name__ == "__main__":
    main()
