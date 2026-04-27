#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

from src.config import NLP_DATA_PATH, NLP_MIN_EXAMPLES, NLP_MODEL_PATH, NLP_REPORT_PATH
from src.nlp_compatibility import ensure_parent_dir, extract_macro_metrics, train_text_compatibility_model
from src.nlp_data_collector import get_nlp_stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train or retrain the pretrained text compatibility NLP model"
    )
    parser.add_argument("--min-examples", type=int, default=NLP_MIN_EXAMPLES)
    parser.add_argument("--data-path", type=str, default=NLP_DATA_PATH, help="Collected real user interactions CSV")
    parser.add_argument("--output-model", type=str, default=NLP_MODEL_PATH)
    parser.add_argument("--output-report", type=str, default=NLP_REPORT_PATH)
    parser.add_argument("--force", action="store_true", help="Train even if real examples are below the threshold")
    parser.add_argument("--synthetic-train", type=str, default="exports/nlp_dataset_train.csv")
    parser.add_argument("--synthetic-val", type=str, default="exports/nlp_dataset_val.csv")
    parser.add_argument("--synthetic-test", type=str, default="exports/nlp_dataset_test.csv")
    parser.add_argument("--real-weight", type=int, default=4, help="Weight multiplier for collected real examples")
    args = parser.parse_args()

    synthetic_available = os.path.exists(args.synthetic_train)
    real_available = os.path.exists(args.data_path)

    if not synthetic_available and not real_available:
        print("No training data found.")
        print("Generate synthetic data first or run the bot to collect real user interactions.")
        return 1

    stats = get_nlp_stats(args.data_path) if real_available else {
        "total": 0,
        "positive": 0,
        "neutral": 0,
        "negative": 0,
        "ready": False,
    }
    print("Real data statistics:")
    print(f"  total:    {stats['total']}")
    print(f"  positive: {stats['positive']}")
    print(f"  neutral:  {stats['neutral']}")
    print(f"  negative: {stats['negative']}")
    print()

    if not synthetic_available and stats["total"] < args.min_examples and not args.force:
        print(f"Not enough real examples: {stats['total']} / {args.min_examples}.")
        print("Use --force or add synthetic data.")
        return 0

    ensure_parent_dir(args.output_model)
    ensure_parent_dir(args.output_report)

    base_dataset = args.synthetic_train if synthetic_available else args.data_path
    validation_dataset = args.synthetic_val if synthetic_available and os.path.exists(args.synthetic_val) else None
    test_dataset = args.synthetic_test if synthetic_available and os.path.exists(args.synthetic_test) else None
    extra_datasets = [args.data_path] if synthetic_available and real_available else None

    print(f"Base dataset: {base_dataset}")
    if extra_datasets:
        print(f"Additional real dataset: {args.data_path} (weight x{args.real_weight})")
    print("Training...")

    report_payload = train_text_compatibility_model(
        dataset_csv_path=base_dataset,
        model_output_path=args.output_model,
        report_output_path=args.output_report,
        validation_csv_path=validation_dataset,
        test_csv_path=test_dataset,
        extra_dataset_paths=extra_datasets,
        real_data_weight=args.real_weight,
    )

    precision, recall, f1 = extract_macro_metrics(report_payload)
    print("Training finished.")
    print(f"Accuracy: {float(report_payload.get('accuracy', 0.0)):.2%}")
    print(f"Macro precision: {precision:.2%}")
    print(f"Macro recall: {recall:.2%}")
    print(f"Macro F1: {f1:.2%}")
    if "final_test_accuracy" in report_payload:
        print(f"Final test accuracy: {float(report_payload['final_test_accuracy']):.2%}")
        print(f"Final test macro F1: {float(report_payload['final_test_macro_f1']):.2%}")
    print(f"Model saved: {args.output_model}")
    print(f"Report saved: {args.output_report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
