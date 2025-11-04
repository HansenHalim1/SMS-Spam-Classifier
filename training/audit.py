from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from training.train import DEFAULT_DATASETS, clean_text, load_dataset
else:
    from .train import DEFAULT_DATASETS, clean_text, load_dataset


def load_combined_frame(datasets: list[str], cache_dir: Path) -> pd.DataFrame:
    frames = [load_dataset(slug, cache_dir) for slug in datasets]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df = df.drop_duplicates(subset="text")
    return df


def audit(datasets: list[str]) -> None:
    cache_dir = Path("./training/data")
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = load_combined_frame(datasets, cache_dir)

    print("Class counts:", Counter(df["label"]))

    texts = df["text"]
    labels = df["label"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Inspect empty vectors with the cleaning function used in training.
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_df=0.95)
    Xt = vectorizer.fit_transform(X_train.map(clean_text))
    Xv = vectorizer.transform(X_valid.map(clean_text))
    empty_train = float((Xt.sum(axis=1) == 0).mean())
    empty_valid = float((Xv.sum(axis=1) == 0).mean())
    print(f"Empty TF-IDF rows -> train: {empty_train:.4f}, valid: {empty_valid:.4f}")

    majority = Counter(y_train).most_common(1)[0][0]
    print("Majority class baseline prediction:", majority)

    logit = make_pipeline(
        TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_df=0.95),
        LogisticRegression(max_iter=500, class_weight="balanced"),
    )
    logit.fit(X_train, y_train)
    preds_lr = logit.predict(X_valid)
    print("LogisticRegression report:\n", classification_report(y_valid, preds_lr, digits=4))
    print("Confusion matrix (ham, spam):\n", confusion_matrix(y_valid, preds_lr, labels=["ham", "spam"]))

    svm = make_pipeline(
        TfidfVectorizer(min_df=2, ngram_range=(1, 2), max_df=0.95),
        LinearSVC(class_weight="balanced"),
    )
    svm.fit(X_train, y_train)
    preds_svm = svm.predict(X_valid)
    print("LinearSVC report:\n", classification_report(y_valid, preds_svm, digits=4))
    print("Confusion matrix (ham, spam):\n", confusion_matrix(y_valid, preds_svm, labels=["ham", "spam"]))

    # Threshold exploration for logistic regression probabilities.
    proba = logit.predict_proba(X_valid)[:, list(logit.classes_).index("spam")]
    thresholds = np.linspace(0.2, 0.8, 61)
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in thresholds:
        preds = np.where(proba >= threshold, "spam", "ham")
        score = f1_score(y_valid, preds, pos_label="spam")
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = score
    print(f"Best probability threshold on validation set: {best_threshold:.2f} (F1={best_f1:.4f})")


def main() -> None:
    audit(DEFAULT_DATASETS)


if __name__ == "__main__":
    main()
