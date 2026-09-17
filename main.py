"""CLI for the Jailbreak Detection MVP.

Usage:
    python main.py train
    python main.py train --data data/dataset_combined.csv
    python main.py train --no-reviewed        # ignore reviewed feedback examples
    python main.py predict "some prompt text"
    python main.py predict --stdin
    python main.py evaluate
    python main.py review                      # interactive human feedback loop

By default, train/evaluate use data/dataset.csv (the small synthetic
starter set). Run `python scripts/fetch_external_datasets.py` first and
pass `--data data/dataset_combined.csv` to train on the merged dataset
that includes the external Hugging Face sources.

Feedback loop: `train` automatically merges any human-reviewed examples
from data/reviewed_examples.csv (created via `python main.py review`) into
the base dataset before training -- but ONLY examples a human has
confirmed. The detector never trains on its own unverified predictions.
Use --no-reviewed to train on the base dataset alone (e.g. for a
before/after comparison).
"""

import argparse
import json
import sys

from src.data import load_dataset, make_splits, validate_and_normalize
from src.evaluate import evaluate_model, print_evaluation, save_evaluation
from src.feedback import merge_with_base_dataset, save_reviewed_example
from src.model import MODEL_PATH, load_model, save_model, train_model
from src.pipeline import analyze_text

DEFAULT_DATA_PATH = "data/dataset.csv"
RESULTS_PATH = "results/evaluation.json"


def _load_training_dataframe(args: argparse.Namespace):
    if getattr(args, "no_reviewed", False):
        return load_dataset(args.data)

    merged = merge_with_base_dataset(args.data)
    return validate_and_normalize(merged)


def cmd_train(args: argparse.Namespace) -> None:
    df = _load_training_dataframe(args)
    splits = make_splits(df)

    pipeline = train_model(splits.x_train, splits.y_train)
    save_model(pipeline, MODEL_PATH)

    source_note = args.data if getattr(args, "no_reviewed", False) else f"{args.data} + reviewed feedback"
    print(f"Trained on {len(splits.x_train)} examples "
          f"(val={len(splits.x_val)}, test={len(splits.x_test)}) from {source_note}.")
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


def cmd_review(args: argparse.Namespace) -> None:
    """Interactive, human-in-the-loop feedback session.

    Workflow: enter a prompt -> see the prediction/reasons -> confirm or
    correct the label -> save as a reviewed example. The model is NOT
    retrained here -- run `python main.py train` afterward to incorporate
    reviewed examples. This separation is deliberate: it prevents the
    detector from ever updating itself based on its own unverified guess.
    """
    pipeline = load_model(MODEL_PATH)
    print("Feedback review session. Press Enter on an empty prompt to stop.\n")

    saved_count = 0
    while True:
        text = input("Prompt to review: ").strip()
        if not text:
            break

        result = analyze_text(pipeline, text)
        print(f"  Prediction: {result.classification.upper()}  (risk={result.risk_score:.2f})")
        print("  Reasons:")
        for reason in result.reasons:
            print(f"    - {reason}")

        label = input("  Confirm actual label [normal/suspicious/skip]: ").strip().lower()
        if label in ("", "skip"):
            print("  Skipped.\n")
            continue
        if label not in ("normal", "suspicious"):
            print(f"  '{label}' is not a valid label (expected 'normal' or 'suspicious'); skipping.\n")
            continue

        save_reviewed_example(text, label, source="manual_review")
        saved_count += 1
        print(f"  Saved as reviewed example ({label}).\n")

    print(f"Session complete. {saved_count} example(s) saved to data/reviewed_examples.csv.")
    if saved_count:
        print("Run `python main.py train` to retrain including these reviewed examples.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jailbreak Detection MVP")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train the model on a dataset CSV")
    train_parser.add_argument(
        "--data",
        default=DEFAULT_DATA_PATH,
        help=f"Path to the base training CSV (default: {DEFAULT_DATA_PATH}). "
        "Use data/dataset_combined.csv after running scripts/fetch_external_datasets.py.",
    )
    train_parser.add_argument(
        "--no-reviewed",
        action="store_true",
        help="Do not merge in data/reviewed_examples.csv even if it exists.",
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

    review_parser = subparsers.add_parser(
        "review", help="Interactive human feedback loop: confirm labels for new examples"
    )
    review_parser.set_defaults(func=cmd_review)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
