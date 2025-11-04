from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from kaggle import api as kaggle_api
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, train_test_split

CLEAN_URL = re.compile(r"https?://\S+")
CLEAN_NUM = re.compile(r"\b\d+\b")
CLEAN_CURRENCY = re.compile(r"[$€£¥]+")
CLEAN_MULTI_SPACE = re.compile(r"\s+")

DEFAULT_DATASETS = [
    "tinu10kumar/sms-spam-dataset",
    "gevabriel/indonesian-sms-spam",
]

SPAM_TOKENS = {
    "spam",
    "spm",
    "junk",
    "promo",
    "iklan",
    "penipuan",
    "fraud",
    "hoax",
    "phishing",
    "1",
    "y",
    "yes",
    "ya",
}
HAM_TOKENS = {
    "ham",
    "legit",
    "normal",
    "bukan spam",
    "bukanspam",
    "non spam",
    "nonspam",
    "not spam",
    "tidak spam",
    "tidakspam",
    "0",
    "n",
    "no",
    "tidak",
    "false",
}
NEGATION_TOKENS = {"not", "non", "no", "bukan", "tidak"}


def clean_text(text: str) -> str:
    text = text or ""
    text = text.lower()
    text = CLEAN_URL.sub(" url ", text)
    text = CLEAN_CURRENCY.sub(" currency ", text)
    text = CLEAN_NUM.sub(" num ", text)
    text = CLEAN_MULTI_SPACE.sub(" ", text)
    return text.strip()


def dataset_cache_dir(base_dir: Path, slug: str) -> Path:
    safe_name = slug.replace("/", "__")
    target = base_dir / safe_name
    target.mkdir(parents=True, exist_ok=True)
    return target


def download_dataset(slug: str, target_dir: Path) -> None:
    kaggle_api.dataset_download_files(slug, path=str(target_dir), unzip=True, quiet=False)


def discover_data_file(dataset_dir: Path) -> Path:
    candidates = [
        path
        for path in dataset_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".txt"}
    ]
    if not candidates:
        raise FileNotFoundError(f"No tabular file (.csv/.tsv/.txt) found in {dataset_dir}")
    candidates.sort(key=lambda p: (len(p.parts), p.name))
    return candidates[0]


def read_table(path: Path) -> pd.DataFrame:
    attempts = [
        ("utf-8", {"sep": ","}),
        ("utf-8", {"sep": None, "engine": "python"}),
        ("latin1", {"sep": ","}),
        ("latin1", {"sep": None, "engine": "python"}),
    ]
    for encoding, kwargs in attempts:
        try:
            return pd.read_csv(path, encoding=encoding, **kwargs)
        except Exception:
            continue
    raise ValueError(f"Unable to parse {path}")


def map_label(raw_value: object) -> str:
    value = str(raw_value or "").strip().lower()
    value_no_space = value.replace(" ", "")
    if not value:
        return "ham"
    if value in HAM_TOKENS or value_no_space in HAM_TOKENS:
        return "ham"
    if value in SPAM_TOKENS or value_no_space in SPAM_TOKENS:
        return "spam"
    if value.isdigit():
        return "spam" if value in {"1", "2", "3"} else "ham"
    if any(token in value for token in NEGATION_TOKENS) and "spam" in value:
        return "ham"
    if any(keyword in value for keyword in ["spam", "promo", "iklan", "penipuan", "fraud", "promo"]):
        return "spam"
    return "ham"


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [str(col).strip().lower() for col in df.columns]
    df.columns = cols

    label_candidates = [
        col
        for col in cols
        if any(token in col for token in ["label", "class", "category", "status", "spam"])
    ]
    text_candidates = [
        col
        for col in cols
        if any(token in col for token in ["text", "message", "sms", "content", "body"])
    ]

    label_col = label_candidates[0] if label_candidates else cols[0]
    text_col = text_candidates[0] if text_candidates else (cols[1] if len(cols) > 1 else cols[0])

    slim = df[[label_col, text_col]].rename(columns={label_col: "label", text_col: "text"})
    slim["label"] = slim["label"].map(map_label)
    slim["text"] = slim["text"].astype(str)
    return slim.dropna(subset=["label", "text"])


def load_dataset(slug: str, base_dir: Path) -> pd.DataFrame:
    print(f"Downloading dataset: {slug}")
    target_dir = dataset_cache_dir(base_dir, slug)
    download_dataset(slug, target_dir)
    data_path = discover_data_file(target_dir)
    df_raw = read_table(data_path)
    df = normalise_columns(df_raw)
    df["source"] = slug
    print(f"Loaded {df.shape[0]} rows from {slug}")
    return df


def vectorize_and_train(
    df: pd.DataFrame,
    vectorizer_params: dict,
    C: float,
    random_state: int = 42,
):
    texts = df["text"].map(clean_text).to_numpy()
    labels = (df["label"] == "spam").astype(int).to_numpy()

    vectorizer = TfidfVectorizer(
        ngram_range=vectorizer_params["ngram_range"],
        min_df=vectorizer_params["min_df"],
        max_df=vectorizer_params["max_df"],
        max_features=vectorizer_params["max_features"],
        sublinear_tf=True,
        norm="l2",
        strip_accents="unicode",
        lowercase=True,
        token_pattern=r"[a-z']+",
    )

    X_train_texts, X_valid_texts, y_train, y_valid = train_test_split(
        texts,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )

    X_train_vec = vectorizer.fit_transform(X_train_texts)
    X_valid_vec = vectorizer.transform(X_valid_texts)

    model = LogisticRegression(
        max_iter=400,
        C=C,
        solver="liblinear",
        class_weight="balanced",
        random_state=random_state,
    )
    start = time.time()
    model.fit(X_train_vec, y_train)
    runtime = time.time() - start

    y_pred_default = model.predict(X_valid_vec)
    f1_default = f1_score(y_valid, y_pred_default)
    print(f"Holdout F1 @0.50: {f1_default:.4f} (training {runtime:.2f}s)")

    probas = model.predict_proba(X_valid_vec)[:, 1]
    threshold_grid = np.linspace(0.1, 0.9, 81)
    best_f1_threshold = 0.5
    best_f1 = f1_default
    precision_target = 0.98
    high_precision_choice: tuple[float, float, float, float] | None = None

    for threshold in threshold_grid:
        preds = (probas >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_valid, preds, pos_label=1, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_f1_threshold = float(threshold)
        if precision >= precision_target:
            if high_precision_choice is None or precision > high_precision_choice[1]:
                high_precision_choice = (float(threshold), precision, recall, f1)

    if high_precision_choice:
        best_threshold = high_precision_choice[0]
        print(
            f"Selected high-precision threshold {best_threshold:.2f} (precision={high_precision_choice[1]:.3f}, "
            f"recall={high_precision_choice[2]:.3f}, f1={high_precision_choice[3]:.4f})"
        )
    else:
        best_threshold = best_f1_threshold
        print(f"Selected F1-optimal threshold {best_threshold:.2f} (F1={best_f1:.4f})")

    print(classification_report(y_valid, (probas >= best_threshold).astype(int), digits=4))

    return vectorizer, model, best_threshold, best_f1


def cross_validate(
    texts: np.ndarray,
    labels: np.ndarray,
    vectorizer_params: dict,
    C: float,
    random_state: int,
) -> float:
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scores: List[float] = []

    for train_idx, valid_idx in splitter.split(texts, labels):
        vect = TfidfVectorizer(
            ngram_range=vectorizer_params["ngram_range"],
            min_df=vectorizer_params["min_df"],
            max_df=vectorizer_params["max_df"],
            max_features=vectorizer_params["max_features"],
            sublinear_tf=True,
            norm="l2",
            strip_accents="unicode",
            lowercase=True,
            token_pattern=r"[a-z']+",
        )

        X_train = vect.fit_transform(texts[train_idx])
        X_valid = vect.transform(texts[valid_idx])

        clf = LogisticRegression(
            max_iter=400,
            C=C,
            solver="liblinear",
            class_weight="balanced",
            random_state=random_state,
        )
        clf.fit(X_train, labels[train_idx])
        scores.append(f1_score(labels[valid_idx], clf.predict(X_valid)))

    return float(np.mean(scores))


def hyperparameter_search(df: pd.DataFrame, random_state: int) -> tuple[dict, float, float]:
    texts = df["text"].map(clean_text).to_numpy()
    labels = (df["label"] == "spam").astype(int).to_numpy()

    param_grid = [
        {"ngram_range": (1, 2), "min_df": 1, "max_df": 0.98, "max_features": 20000, "C": 1.5},
        {"ngram_range": (1, 3), "min_df": 1, "max_df": 0.97, "max_features": 30000, "C": 2.0},
        {"ngram_range": (1, 4), "min_df": 1, "max_df": 0.96, "max_features": 32000, "C": 2.5},
        {"ngram_range": (1, 4), "min_df": 2, "max_df": 0.95, "max_features": 28000, "C": 3.0},
    ]

    best_score = -1.0
    best_params: dict | None = None

    for params in param_grid:
        score = cross_validate(texts, labels, params, params["C"], random_state)
        print(
            f"CV F1={score:.4f} | ngram={params['ngram_range']} min_df={params['min_df']} "
            f"max_df={params['max_df']} max_features={params['max_features']} C={params['C']}"
        )
        if score > best_score:
            best_score = score
            best_params = params.copy()

    if best_params is None:
        raise RuntimeError("Hyperparameter search failed to evaluate any configuration.")

    best_C = best_params.pop("C")
    return best_params, best_C, best_score


def save_artifacts(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    threshold: float,
    outdir: Path,
    version: str = "v1.1.0",
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    vocab = {token: int(idx) for token, idx in vectorizer.vocabulary_.items()}
    idf = vectorizer.idf_.tolist()
    coef = model.coef_.reshape(-1).tolist()
    intercept = float(model.intercept_[0])

    model_json = {
        "model_type": "logreg",
        "version": version,
        "threshold": threshold,
        "classes": ["ham", "spam"],
        "coef": coef,
        "intercept": intercept,
        "vocabulary": vocab,
        "idf": idf,
        "tfidf": {
            "ngram_range": vectorizer.ngram_range,
            "sublinear_tf": True,
            "norm": "l2",
            "use_idf": True,
            "min_df": vectorizer.min_df,
            "max_df": float(vectorizer.max_df)
            if hasattr(vectorizer.max_df, "__float__")
            else vectorizer.max_df,
            "max_features": vectorizer.max_features,
            "lowercase": True,
            "strip_accents": "unicode",
            "token_pattern": vectorizer.token_pattern,
        },
        "training_meta": {
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": model.max_iter,
        },
    }
    (outdir / "model.pkl").write_text(json.dumps(model_json))

    feature_config = {
        "token_pattern": r"[a-zA-Z']+",
        "lowercase": True,
        "strip": True,
    }
    (outdir / "feature_config.json").write_text(json.dumps(feature_config, indent=2))

    label_map = {"0": "ham", "1": "spam"}
    (outdir / "label_map.json").write_text(json.dumps(label_map, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        help="Kaggle dataset slug to include (repeat flag to merge multiple datasets).",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=Path("../artifacts"))
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DEFAULT_DATASETS
    print("Datasets:")
    for slug in datasets:
        print(f" - {slug}")

    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for slug in datasets:
        df = load_dataset(slug, data_dir)
        frames.append(df)

    dataset = pd.concat(frames, ignore_index=True)
    dataset = dataset.drop_duplicates(subset="text")
    print(f"Combined dataset shape: {dataset.shape}")

    best_vec_params, best_C, cv_score = hyperparameter_search(dataset, args.random_state)
    print(
        "Selected params -> "
        f"ngram_range={best_vec_params['ngram_range']}, "
        f"min_df={best_vec_params['min_df']}, max_df={best_vec_params['max_df']}, "
        f"max_features={best_vec_params['max_features']}, C={best_C} "
        f"(CV F1={cv_score:.4f})"
    )

    vectorizer, model, threshold, holdout_f1 = vectorize_and_train(
        dataset,
        vectorizer_params=best_vec_params,
        C=best_C,
        random_state=args.random_state,
    )

    print(f"Final holdout F1 @best threshold: {holdout_f1:.4f}")

    save_artifacts(vectorizer, model, threshold, args.outdir, version="v1.1.0")


if __name__ == "__main__":
    main()
