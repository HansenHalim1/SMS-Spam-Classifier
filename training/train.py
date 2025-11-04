import argparse
import json
import re
import time
from pathlib import Path

import pandas as pd
from kaggle import api as kaggle_api
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

CLEAN_URL = re.compile(r"https?://\S+")
CLEAN_NUM = re.compile(r"\b\d+\b")


def clean_text(s: str) -> str:
    s = s or ""
    s = s.lower()
    s = CLEAN_URL.sub(" url ", s)
    s = CLEAN_NUM.sub(" num ", s)
    return s.strip()


def load_kaggle_dataset(slug: str, outdir: Path) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    kaggle_api.dataset_download_files(slug, path=str(outdir), unzip=True, quiet=False)

    candidates = [
        outdir / "spam.csv",
        outdir / "SMSSpamCollection.csv",
        outdir / "SMSSpamCollection",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    for path in outdir.glob("**/*"):
        if path.suffix.lower() in {".csv", ".tsv", ""}:
            return path

    raise FileNotFoundError("Could not locate dataset file after download.")


def read_dataset(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding="latin1")
    except Exception:
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except Exception:
            df = pd.read_csv(path, sep="\t", header=None, names=["label", "text"])

    cols = [c.lower() for c in df.columns]
    df.columns = cols

    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})

    if "label" not in df.columns or "text" not in df.columns:
        df = df.rename(columns={cols[0]: "label", cols[1]: "text"})

    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].map(lambda x: "spam" if str(x).strip().lower() == "spam" else "ham")
    return df


def vectorize_and_train(df: pd.DataFrame, ngram=(1, 2), C=2.0, random_state=42):
    X = df["text"].map(clean_text).tolist()
    y = (df["label"] == "spam").astype(int).values

    vectorizer = TfidfVectorizer(
        ngram_range=ngram,
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        norm="l2",
    )
    X_vec = vectorizer.fit_transform(X)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_vec,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    model = LogisticRegression(
        max_iter=200,
        C=C,
        solver="liblinear",
        random_state=random_state,
    )
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_valid)
    f1 = f1_score(y_valid, y_pred)
    print(f"Train time: {train_time:.2f}s")
    print("F1:", f1)
    print(classification_report(y_valid, y_pred, digits=4))

    return vectorizer, model, f1


def save_artifacts(vectorizer: TfidfVectorizer, model: LogisticRegression, outdir: Path, version="v1.0.0") -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    vocab = {token: int(idx) for token, idx in vectorizer.vocabulary_.items()}
    idf = vectorizer.idf_.tolist()
    coef = model.coef_.reshape(-1).tolist()
    intercept = float(model.intercept_[0])

    model_json = {
        "model_type": "logreg",
        "version": version,
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
            "lowercase": True,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="uciml/sms-spam-collection-dataset")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--outdir", type=Path, default=Path("../artifacts"))
    args = parser.parse_args()

    data_dir = Path("./data")
    data_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = load_kaggle_dataset(args.dataset, data_dir)
    df = read_dataset(dataset_path)
    print("Loaded:", df.shape)

    _, _, f1_baseline = vectorize_and_train(df, ngram=(1, 1), C=1.0, random_state=args.random_state)
    vectorizer, model, f1_improved = vectorize_and_train(
        df,
        ngram=(1, 2),
        C=2.0,
        random_state=args.random_state,
    )

    print(f"Baseline F1 ~ {f1_baseline:.3f} -> Improved F1 {f1_improved:.3f}")

    save_artifacts(vectorizer, model, args.outdir, version="v1.0.0")


if __name__ == "__main__":
    main()
