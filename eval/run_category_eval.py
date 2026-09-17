"""Category-level evaluation on a held-out set that is NOT used for
training (see eval/category_eval_set.csv). This is separate from
`python main.py evaluate` (which re-splits the training CSV) -- this
script measures generalization to genuinely unseen examples across the
specific categories requested during debugging: direct/indirect synthetic
manipulation, encoded/obfuscated variants, mixed benign+suspicious
content, and previously-unseen wording.

Usage:
    python eval/run_category_eval.py
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model import MODEL_PATH, load_model
from src.pipeline import analyze_batch

EVAL_SET_PATH = Path(__file__).parent / "category_eval_set.csv"
RESULTS_PATH = Path(__file__).parent / "category_eval_results.csv"


def main() -> None:
    pipeline = load_model(MODEL_PATH)
    df = pd.read_csv(EVAL_SET_PATH)

    results = analyze_batch(pipeline, df["text"].tolist())
    df["predicted"] = [r.classification for r in results]
    df["risk_score"] = [round(r.risk_score, 3) for r in results]
    df["correct"] = df["label"] == df["predicted"]

    print("=== Overall ===")
    overall_acc = df["correct"].mean()
    print(f"Accuracy: {overall_acc:.3f} ({df['correct'].sum()}/{len(df)})")

    p, r, f1, _ = precision_recall_fscore_support(
        df["label"], df["predicted"], labels=["normal", "suspicious"], zero_division=0
    )
    print(f"Precision(suspicious)={p[1]:.3f} Recall(suspicious)={r[1]:.3f} F1(suspicious)={f1[1]:.3f}")
    cm = confusion_matrix(df["label"], df["predicted"], labels=["normal", "suspicious"])
    print("Confusion matrix [normal, suspicious]:", cm.tolist())

    print("\n=== By category ===")
    for cat, group in df.groupby("category"):
        acc = group["correct"].mean()
        print(f"{cat:22s} n={len(group):2d} accuracy={acc:.2f}")

    print("\n=== Misclassifications ===")
    wrong = df[~df["correct"]]
    if wrong.empty:
        print("(none)")
    for _, row in wrong.iterrows():
        print(f"[{row['category']}] expected={row['label']} got={row['predicted']} risk={row['risk_score']}: {row['text'][:80]}")

    df.to_csv(RESULTS_PATH, index=False)
    print(f"\nFull per-example results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
