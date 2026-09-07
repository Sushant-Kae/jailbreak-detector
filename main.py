"""CLI for the Jailbreak Detection MVP.

Usage:
    python main.py train
    python main.py train --data data/dataset_combined.csv
    python main.py predict "some prompt text"
    python main.py predict --stdin
    python main.py evaluate

By default, train/evaluate use data/dataset.csv (the small synthetic
starter set). Run `python scripts/fetch_external_datasets.py` first and
pass `--data data/dataset_combined.csv` to train on the merged dataset
that includes the external Hugging Face sources.
"""

import argparse
import json
import sys

from src.data import load_dataset, make_splits
from src.evaluate import evaluate_model, print_evaluation, save_evaluation
from src.model import MODEL_PATH, load_model, save_model, train_model
from src.pipeline import analyze_text

DEFAULT_DATA_PATH = "data/dataset.csv"
RESULTS_PATH = "results/evaluation.json"


def cmd_train(args: argparse.Namespace) -> None:
    df = load_dataset(args.data)
    splits = make_splits(df)

    pipeline = train_model(splits.x_train, splits.y_train)
    save_model(pipeline, MODEL_PATH)

    print(f"Trained on {len(splits.x_train)} examples "
          f"(val={len(splits.x_val)}, test={len(splits.x_test)}) from {args.data}.")
    print(f"Model saved to {MODEL_PATH}")

    # Quick sanity check on validation split.
    val_results = evaluate_model(pipeline, splits.x_val, splits.y_val)
    print("\nValidation performance:")
    print_evaluation(val_results)


def cmd_predict(args: argparse.Namespace) -> None:
    pipeline = load_model(MODEL_PATH)

    if args.stdin:
        text = sys.stdin.read()
    else:
        text = args.text

    result = analyze_text(pipeline, text)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
        return

    print(f"Classification: {result.classification.upper()}")
    print(f"Risk Score: {result.risk_score:.2f}")
    print("Reasons:")
    for reason in result.reasons:
        print(f"- {reason}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    pipeline = load_model(MODEL_PATH)
    df = load_dataset(args.data)
    splits = make_splits(df)

    results = evaluate_model(pipeline, splits.x_test, splits.y_test)
    print_evaluation(results)
    save_evaluation(results, RESULTS_PATH)
    print(f"\nFull results saved to {RESULTS_PATH}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jailbreak Detection MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model on a dataset CSV")
    train_parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to the training CSV (default: {DEFAULT_DATA_PATH}). "
        "Use data/dataset_combined.csv after running scripts/fetch_external_datasets.py.",
    )
    train_parser.set_defaults(func=cmd_train)

    predict_parser = subparsers.add_parser("predict", help="Classify a single prompt")
    predict_parser.add_argument("text", nargs="?", default="", help="Prompt text to analyze")
    predict_parser.add_argument("--stdin", action="store_true", help="Read text from stdin")
    predict_parser.add_argument("--json", action="store_true", help="Output as JSON")
    predict_parser.set_defaults(func=cmd_predict)

    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate the model on the test split")
    evaluate_parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help="Path to the CSV to re-split and evaluate against "
        "(should match what the model was trained with).",
    )
    evaluate_parser.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
