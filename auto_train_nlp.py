#!/usr/bin/env python3
"""
Скрипт для автоматического мониторинга и переобучения NLP модели.

Запускает периодические проверки и автоматически переобучивает модель
когда достаточно данных и есть деградация качества.

Использование:
    # Запустить один раз
    python auto_train_nlp.py

    # Добавить в cron (Linux/Mac)
    0 */6 * * * cd /path/to/project && python auto_train_nlp.py

    # Добавить в Task Scheduler (Windows)
    # Запланировать: auto_train_nlp.py каждые 6 часов
"""

import argparse
import logging
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from src.config import NLP_DATA_PATH, NLP_MIN_EXAMPLES, NLP_MODEL_PATH, NLP_REPORT_PATH
from src.nlp_compatibility import extract_macro_metrics, train_text_compatibility_model
from src.nlp_data_collector import get_nlp_stats
from src.nlp_metrics import NLPMetricsTracker

# Setup logging
Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/nlp_auto_train.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


class AutoTrainer:
    # Инициализирует объект и сохраняет нужные зависимости.
    def __init__(
        self,
        data_path=NLP_DATA_PATH,
        model_path=NLP_MODEL_PATH,
        min_examples=NLP_MIN_EXAMPLES,
    ):
        self.data_path = data_path
        self.model_path = model_path
        self.min_examples = min_examples
        self.tracker = NLPMetricsTracker()

    # Возвращает текущую точность модели по накопленным метрикам.
    def get_current_accuracy(self):
        """Get current model accuracy"""
        try:
            metrics = self.tracker.calculate_metrics(hours=None)
            return metrics.get("accuracy", 0)
        except Exception as e:
            logger.warning(f"Не удалось получить текущую точность: {e}")
            return 0

    # Решает, нужно ли запускать переобучение NLP-модели.
    def should_train(self):
        """Determine if we should train"""
        # Check if data file exists
        if not os.path.exists(self.data_path):
            logger.info("Файл данных пока не найден")
            return False

        # Get data statistics
        try:
            stats = get_nlp_stats(self.data_path)
            total = stats["total"]

            if total < self.min_examples:
                logger.info(
                    f"Недостаточно примеров: {total}/{self.min_examples}"
                )
                return False

            # Check if accuracy has degraded
            current_accuracy = self.get_current_accuracy()

            # If no previous accuracy, train anyway
            if current_accuracy == 0:
                logger.info(f"Есть {total} примеров, базовой точности нет — пора обучать")
                return True

            # If accuracy is good, don't retrain frequently
            if current_accuracy > 0.75:
                logger.info(
                    f"Текущая точность хорошая ({current_accuracy:.1%}), обучение пропускается"
                )
                return False

            # If accuracy is low, train
            logger.info(
                f"Текущая точность низкая ({current_accuracy:.1%}), требуется обучение"
            )
            return True

        except Exception as e:
            logger.error(f"Ошибка проверки условий обучения: {e}")
            return False

    # Запускает обучение NLP-модели и сохраняет результат.
    def run_training(self):
        """Execute training"""
        logger.info("=" * 60)
        logger.info("Запуск обучения NLP-модели")
        logger.info("=" * 60)

        try:
            stats = get_nlp_stats(self.data_path)
            logger.info(f"Данные для обучения: {stats['total']} примеров")
            logger.info(
                f"  Положительных: {stats['positive']}, Нейтральных: {stats['neutral']}, "
                f"Отрицательных: {stats['negative']}"
            )

            # Backup old model
            if os.path.exists(self.model_path):
                backup_root = Path(self.model_path).parent / "backups"
                backup_root.mkdir(parents=True, exist_ok=True)
                backup_path = backup_root / f"nlp_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
                shutil.copy2(self.model_path, backup_path)
                logger.info(f"Создана резервная копия старой модели: {backup_path}")

            # Create output directories
            model_dir = Path(self.model_path).parent
            if str(model_dir) and str(model_dir) != ".":
                model_dir.mkdir(parents=True, exist_ok=True)
            Path("reports").mkdir(exist_ok=True)

            # Train new model
            logger.info("Идет обучение...")
            report = train_text_compatibility_model(
                dataset_csv_path=self.data_path,
                model_output_path=self.model_path,
                report_output_path=NLP_REPORT_PATH,
            )

            precision, recall, f1 = extract_macro_metrics(report)

            # Log results
            logger.info("Обучение успешно завершено")
            logger.info(f"  Точность (Accuracy): {report.get('accuracy', 0):.1%}")
            logger.info(f"  Точность по положительному классу (Precision): {precision:.1%}")
            logger.info(f"  Полнота (Recall): {recall:.1%}")
            logger.info(f"  F1-мера: {f1:.1%}")

            logger.info("=" * 60)
            return True

        except Exception as e:
            logger.error(f"Ошибка обучения: {e}", exc_info=True)
            return False

    # Запускает основной цикл обработки событий.
    def run(self):
        """Main entry point"""
        logger.info(f"Автотренировщик NLP запущен: {datetime.now()}")

        if self.should_train():
            success = self.run_training()
            return 0 if success else 1
        else:
            logger.info("Сейчас обучение не требуется")
            return 0


# Точка входа: разбирает аргументы и запускает нужный сценарий.
def main():
    parser = argparse.ArgumentParser(
        description="Автоматически обучать NLP-модель, когда она готова"
    )
    parser.add_argument(
        "--data-path",
        default=NLP_DATA_PATH,
        help="Путь к данным для обучения",
    )
    parser.add_argument(
        "--model-path",
        default=NLP_MODEL_PATH,
        help="Путь сохранения модели",
    )
    parser.add_argument(
        "--min-examples",
        type=int,
        default=NLP_MIN_EXAMPLES,
        help="Минимум примеров перед обучением",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Запустить обучение принудительно, даже если условия не выполнены",
    )

    args = parser.parse_args()

    # Create logs directory
    Path("logs").mkdir(exist_ok=True)

    trainer = AutoTrainer(
        data_path=args.data_path,
        model_path=args.model_path,
        min_examples=args.min_examples,
    )

    if args.force:
        logger.info("Принудительный запуск обучения (--force)")
        return 0 if trainer.run_training() else 1

    return trainer.run()


if __name__ == "__main__":
    sys.exit(main())
