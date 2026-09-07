"""Evaluation utilities: metrics + confusion matrix on a held-out split."""

import json
from pathlib import Path

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from src.pipeline import analyze_batch

LABELS = ["normal", "suspicious"]


def evaluate_model(pipeline: Pipeline, x_test: list[str], y_test: list[str]) -> dict:
    """Run the full pipeline (ML + signals + risk) on the test set and
    compute standard classification metrics against the final
    `classification` output (not just the raw ML label)."""
    results = analyze_batch(pipeline, x_test)
    y_pred = [r.classification for r in results]

    report = classification_report(
        y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=LABELS)

    false_positives = [
        text
        for text, true, pred in zip(x_test, y_test, y_pred)
        if true == "normal" and pred == "suspicious"
    ]
    false_negatives = [
        text
        for text, true, pred in zip(x_test, y_test, y_pred)
        if true == "suspicious" and pred == "normal"
    ]

    return {
        "accuracy": report["accuracy"],
        "precision_suspicious": report["suspicious"]["precision"],
        "recall_suspicious": report["suspicious"]["recall"],
        "f1_suspicious": report["suspicious"]["f1-score"],
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": cm.tolist(),
        },
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "full_report": report,
    }


def save_evaluation(results: dict, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def print_evaluation(results: dict) -> None:
    print(f"Accuracy:  {results['accuracy']:.3f}")
    print(f"Precision (suspicious): {results['precision_suspicious']:.3f}")
    print(f"Recall (suspicious):    {results['recall_suspicious']:.3f}")
    print(f"F1 (suspicious):        {results['f1_suspicious']:.3f}")
    print(f"Confusion matrix {results['confusion_matrix']['labels']}:")
    for row in results["confusion_matrix"]["matrix"]:
        print(f"  {row}")
    if results["false_positives"]:
        print(f"False positives ({len(results['false_positives'])}):")
        for t in results["false_positives"]:
            print(f"  - {t}")
    if results["false_negatives"]:
        print(f"False negatives ({len(results['false_negatives'])}):")
        for t in results["false_negatives"]:
            print(f"  - {t}")
