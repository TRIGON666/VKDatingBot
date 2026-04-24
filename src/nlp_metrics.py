"""Мониторинг качества NLP модели в реальном времени."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from typing import Optional


logger = logging.getLogger(__name__)


class NLPMetricsTracker:
    """Отслеживание точности NLP предсказаний."""
    
    def __init__(self, log_file: str = "data/nlp_metrics.jsonl"):
        self.log_file = log_file
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    def log_prediction(
        self,
        viewer_id: int,
        viewed_id: int,
        predicted_score: float,
        actual_action: str,
        model_version: str = "v1",
    ) -> None:
        """
        Логировать одно предсказание.
        
        Args:
            viewer_id: ID юзера
            viewed_id: ID кандидата
            predicted_score: Предсказанная совместимость (0-100%)
            actual_action: Реальное действие пользователя
            model_version: Версия модели
        """
        
        actual_class = "positive" if actual_action == "like" else (
            "negative" if actual_action in {"dislike", "block", "report"} else "neutral"
        )
        
        if predicted_score >= 60:
            predicted_class = "positive"
        elif predicted_score <= 40:
            predicted_class = "negative"
        else:
            predicted_class = "neutral"
        
        record = {
            "timestamp": datetime.now().isoformat(),
            "viewer_id": viewer_id,
            "viewed_id": viewed_id,
            "predicted_score": round(predicted_score, 2),
            "predicted_class": predicted_class,
            "actual_action": actual_action,
            "actual_class": actual_class,
            "model_version": model_version,
            "correct": predicted_class == actual_class,
        }
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except (OSError, TypeError, ValueError):
            # Логирование метрик не должно ломать основную логику бота.
            logger.exception("Failed to write NLP metrics record")
            return
    
    def calculate_metrics(self, hours: Optional[int] = 24) -> dict:
        """
        Рассчитать метрики за последние N часов.
        
        Returns:
            {
                "total_predictions": int,
                "accuracy": float,
                "precision_positive": float,
                "recall_positive": float,
                "period": "24h",
            }
        """
        
        empty_result = {
            "predictions_count": 0,
            "total_predictions": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "precision_positive": 0.0,
            "recall_positive": 0.0,
            "period": "all" if hours is None else f"{hours}h",
        }

        if not os.path.exists(self.log_file):
            return empty_result

        now = datetime.now()
        cutoff = None if hours is None else now - timedelta(hours=hours)
        
        total_correct = 0
        total_predictions = 0
        positive_metrics = {"tp": 0, "fp": 0, "fn": 0}
        
        with open(self.log_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                    ts = datetime.fromisoformat(record["timestamp"])
                    if cutoff is not None and ts < cutoff:
                        continue
                    
                    total_predictions += 1
                    
                    if record["correct"]:
                        total_correct += 1
                    
                    if record["actual_class"] == "positive":
                        if record["predicted_class"] == "positive":
                            positive_metrics["tp"] += 1
                        else:
                            positive_metrics["fn"] += 1
                    elif record["predicted_class"] == "positive":
                        positive_metrics["fp"] += 1
                
                except json.JSONDecodeError:
                    continue
                except (ValueError, KeyError, TypeError):
                    continue
        
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        precision = (
            positive_metrics["tp"] / (positive_metrics["tp"] + positive_metrics["fp"])
            if (positive_metrics["tp"] + positive_metrics["fp"]) > 0 else 0
        )
        recall = (
            positive_metrics["tp"] / (positive_metrics["tp"] + positive_metrics["fn"])
            if (positive_metrics["tp"] + positive_metrics["fn"]) > 0 else 0
        )
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        
        return {
            "predictions_count": total_predictions,
            "total_predictions": total_predictions,
            "accuracy": round(accuracy, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "precision_positive": round(precision, 4),
            "recall_positive": round(recall, 4),
            "period": "all" if hours is None else f"{hours}h",
        }
    
    def export_report(self, filepath: str = "reports/nlp_metrics.json") -> None:
        """Экспортировать отчёт о качестве."""
        
        metrics = self.calculate_metrics(hours=24)
        metrics_7d = self.calculate_metrics(hours=24*7)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "metrics_24h": metrics,
            "metrics_7d": metrics_7d,
        }
        
        report_dir = os.path.dirname(filepath)
        if report_dir:
            os.makedirs(report_dir, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
