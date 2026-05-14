#!/usr/bin/env python3
"""
Утилиты для управления версиями и восстановления NLP модели.

Команды:
    python nlp_utils.py --status                # Проверка статуса
    python nlp_utils.py --backup                # Backup текущей модели
    python nlp_utils.py --restore               # Восстановить из backup
    python nlp_utils.py --cleanup               # Очистить старые файлы
    python nlp_utils.py --reset                 # Переустановить модель по умолчанию
"""

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.config import NLP_DATA_PATH, NLP_MODEL_PATH, NLP_REPORT_PATH


class NLPModelManager:
    # Инициализирует объект и сохраняет нужные зависимости.
    def __init__(self, model_path: str = NLP_MODEL_PATH):
        self.model_path = model_path
        self.backup_dir = str(Path(model_path).parent / "backups")

        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)
    def _list_backup_names(self) -> list[str]:
        if not os.path.exists(self.backup_dir):
            return []
        model_suffix = Path(self.model_path).suffix
        return sorted(
            [f for f in os.listdir(self.backup_dir) if Path(f).suffix in {model_suffix, ".pkl", ".joblib"}],
            reverse=True,
        )

    # Показывает состояние текущей NLP-модели и связанных файлов.
    def status(self):
        """Show model status"""
        print("\n" + "=" * 60)
        print("СТАТУС NLP МОДЕЛИ")
        print("=" * 60)

        if os.path.exists(self.model_path):
            size_mb = os.path.getsize(self.model_path) / 1024 / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(self.model_path))
            print("\nТекущая модель:")
            print(f"  Путь:     {self.model_path}")
            print(f"  Размер:   {size_mb:.2f} MB")
            print(f"  Обновлена:{mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print(f"\nМодель не найдена: {self.model_path}")

        backups = self._list_backup_names()

        if backups:
            print(f"\nРезервные копии ({len(backups)}):")
            for i, backup in enumerate(backups[:5], 1):
                backup_path = os.path.join(self.backup_dir, backup)
                size_mb = os.path.getsize(backup_path) / 1024 / 1024
                mtime = datetime.fromtimestamp(os.path.getmtime(backup_path))
                print(f"  {i}. {backup}")
                print(f"     Размер: {size_mb:.2f} MB, Дата: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("\nРезервные копии не найдены")

        print("\nФайлы данных:")
        data_files = {
            "Данные обучения": NLP_DATA_PATH,
            "Метрики": "data/nlp_metrics.jsonl",
            "Отчет": NLP_REPORT_PATH,
        }

        for name, path in data_files.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / 1024 / 1024
                print(f"  OK  {name}: {path} ({size_mb:.2f} MB)")
            else:
                print(f"  --- {name}: {path} (не найден)")

        print()

    # Создает резервную копию текущей NLP-модели.
    def backup(self):
        """Backup current model"""
        if not os.path.exists(self.model_path):
            print(f"\nМодель не найдена: {self.model_path}")
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = os.path.join(self.backup_dir, f"text_compatibility_{timestamp}{Path(self.model_path).suffix}")

        try:
            shutil.copy2(self.model_path, backup_path)
            size_mb = os.path.getsize(backup_path) / 1024 / 1024
            print(f"\nРезервная копия создана: {backup_path}")
            print(f"Размер: {size_mb:.2f} MB")
            return True
        except Exception as e:
            print(f"\nОшибка создания копии: {e}")
            return False

    # Восстанавливает NLP-модель из резервной копии.
    def restore(self, backup_file: Optional[str] = None):
        """Restore from backup"""
        if backup_file:
            backup_path = os.path.join(self.backup_dir, backup_file)
        else:
            backups = self._list_backup_names()
            if not backups:
                print("\nРезервные копии не найдены")
                return False

            backup_path = os.path.join(self.backup_dir, backups[0])

        if not os.path.exists(backup_path):
            print(f"\nКопия не найдена: {backup_path}")
            return False

        try:
            if os.path.exists(self.model_path):
                self.backup()

            shutil.copy2(backup_path, self.model_path)
            print(f"\nВосстановлено из: {backup_path}")
            print(f"Текущая модель: {self.model_path}")
            return True
        except Exception as e:
            print(f"\nОшибка восстановления: {e}")
            return False

    # Удаляет старые резервные копии модели.
    def cleanup(self):
        """Remove old backups (keep last 5)"""
        backups = self._list_backup_names()

        if len(backups) <= 5:
            print(f"\nОчистка не нужна: копий {len(backups)}, оставляем 5")
            return

        to_delete = backups[5:]
        deleted = 0

        for backup in to_delete:
            backup_path = os.path.join(self.backup_dir, backup)
            try:
                os.remove(backup_path)
                deleted += 1
                print(f"  Удалено: {backup}")
            except Exception as e:
                print(f"  Ошибка удаления {backup}: {e}")

        print(f"\nУдалено {deleted} старых копий, оставлено 5 последних")

    # Удаляет текущую модель после резервного копирования.
    def reset(self):
        """Reset to default training"""
        if os.path.exists(self.model_path):
            if not self.backup():
                print("\nНе удалось создать копию текущей модели")

        if not os.path.exists(self.model_path):
            print("\nМодель уже отсутствует, сброс не требуется")
            return

        try:
            os.remove(self.model_path)
            print(f"\nМодель сброшена: {self.model_path}")
            print("Новая модель будет обучена при наличии данных")
        except Exception as e:
            print(f"\nОшибка сброса: {e}")

    # Показывает доступные версии NLP-модели.
    def list_models(self):
        """List all available model versions"""
        print("\n" + "=" * 60)
        print("ДОСТУПНЫЕ МОДЕЛИ")
        print("=" * 60)

        backups = self._list_backup_names()

        if os.path.exists(self.model_path):
            print("\nТекущая модель (активная):")
            size_mb = os.path.getsize(self.model_path) / 1024 / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(self.model_path))
            print(f"  {self.model_path}")
            print(f"  Размер: {size_mb:.2f} MB")
            print(f"  Дата:   {mtime.strftime('%Y-%m-%d %H:%M:%S')}\n")

        if backups:
            print(f"Резервные копии ({len(backups)}):")
            for i, backup in enumerate(backups[:10], 1):
                backup_path = os.path.join(self.backup_dir, backup)
                size_mb = os.path.getsize(backup_path) / 1024 / 1024
                mtime = datetime.fromtimestamp(os.path.getmtime(backup_path))
                print(f"  {i:2d}. {backup}")
                print(f"      Размер: {size_mb:.2f} MB, Дата: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("Резервные копии не найдены")

        print()


# Точка входа: разбирает аргументы и запускает нужный сценарий.
def main():
    parser = argparse.ArgumentParser(description="Утилиты управления версиями NLP модели")
    parser.add_argument("--status", action="store_true", help="Показать статус модели")
    parser.add_argument("--backup", action="store_true", help="Создать резервную копию")
    parser.add_argument("--restore", nargs="?", const=True, help="Восстановить из копии")
    parser.add_argument("--cleanup", action="store_true", help="Удалить старые копии")
    parser.add_argument("--reset", action="store_true", help="Сбросить текущую модель")
    parser.add_argument("--list", action="store_true", help="Показать список моделей")
    parser.add_argument("--model-path", default=NLP_MODEL_PATH)

    args = parser.parse_args()

    manager = NLPModelManager(model_path=args.model_path)

    if args.status:
        manager.status()
    elif args.backup:
        manager.backup()
    elif args.restore is not None:
        backup_name = args.restore if isinstance(args.restore, str) else None
        manager.restore(backup_name)
    elif args.cleanup:
        manager.cleanup()
    elif args.reset:
        manager.reset()
    elif args.list:
        manager.list_models()
    else:
        # Default to status
        manager.status()


if __name__ == "__main__":
    main()
