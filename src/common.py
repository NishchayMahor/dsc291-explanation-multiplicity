"""COMPAS loading (ProPublica's filters) and binarization, shared by the baseline scripts."""
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
LOCAL = HERE.parent / "data" / "compas-scores-two-years.csv"
URL = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"


def load_compas():
    """ProPublica's COMPAS file, downloaded once into data/ and cached."""
    if not LOCAL.exists():
        print(f"[data] downloading {URL} -> {LOCAL}")
        LOCAL.parent.mkdir(exist_ok=True)
        pd.read_csv(URL).to_csv(LOCAL, index=False)
    df = pd.read_csv(LOCAL)
    df = df[(df.days_b_screening_arrest <= 30) & (df.days_b_screening_arrest >= -30)
            & (df.is_recid != -1) & (df.c_charge_degree != "O") & (df.score_text != "N/A")].copy()
    X = pd.DataFrame({
        "age": df.age,
        "priors": df.priors_count,
        "juv_fel": df.juv_fel_count,
        "juv_misd": df.juv_misd_count,
        "juv_other": df.juv_other_count,
        "male": (df.sex == "Male").astype(int),
        "felony": (df.c_charge_degree == "F").astype(int),
    })
    y = df.two_year_recid.values
    meta = df[["race", "decile_score", "sex", "age"]].reset_index(drop=True)
    return X.reset_index(drop=True), y, meta


def binarize(X):
    """Threshold features so sparse models stay readable (age bands, prior counts)."""
    B = pd.DataFrame(index=X.index)
    for lo, hi in [(18, 20), (21, 23), (24, 25), (26, 30), (31, 40), (41, 200)]:
        B[f"age_{lo}_{hi}"] = ((X.age >= lo) & (X.age <= hi)).astype(int)
    for t in [0, 1, 2, 3, 5, 10]:
        B[f"priors_gt_{t}"] = (X.priors > t).astype(int)
    B["juv_any"] = ((X.juv_fel + X.juv_misd + X.juv_other) > 0).astype(int)
    B["male"] = X.male
    B["felony"] = X.felony
    return B


def splits(X, y, B, seed=1):
    """train / validation / test = 49 / 21 / 30, as in our talk (set chosen on val, flips measured on test)."""
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte, Btr, Bte = train_test_split(X, y, B, test_size=0.3, stratify=y, random_state=seed)
    Xtr, Xva, ytr, yva, Btr, Bva = train_test_split(Xtr, ytr, Btr, test_size=0.3, stratify=ytr, random_state=seed + 1)
    return (Xtr, Xva, Xte), (ytr, yva, yte), (Btr, Bva, Bte)


def pairwise_disagreement(P):
    """Mean share of people on which two rows of a prediction matrix P (models x people) differ."""
    k = len(P)
    if k < 2:
        return 0.0
    d = [(P[i] != P[j]).mean() for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(d))
