"""P3 baseline: do explanations survive retraining? XGBoost + TreeSHAP vs sparse models' own reasons.

R bootstrap retrains of each model class on COMPAS (same hyperparameters, as in our retraining test). On the shared test set:
  - xgboost: per-person TreeSHAP attributions. Uses `shap` if installed, else XGBoost's built-in
    TreeSHAP (Booster.predict(pred_contribs=True)), which computes the same exact Tree SHAP values.
  - sparse_l1: per-person contributions coef_j * x_j on the binarized features (the model's exact reason).
  - tree_d3: per-person set of features on the decision path (the rule that fires).
Metrics per person, averaged over all retrain pairs, then over people:
  - top-k Jaccard of the k most important features (k = 2, 3)
  - Spearman rank correlation of |attribution| (attribution models only)
  - prediction flip rate (did the decision itself change?)
and the same metrics restricted to people whose prediction did NOT flip (explanation change with no outcome change).

Usage: python src/run_baseline.py [--retrains 20] [--k 2 3]
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import DMatrix, XGBClassifier

from common import binarize, load_compas

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--retrains", type=int, default=20)
ap.add_argument("--k", type=int, nargs="*", default=[2, 3])
ap.add_argument("--max-people", type=int, default=600, help="test people used for pairwise metrics (speed)")
args = ap.parse_args()

try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False
    print("[optional] shap not installed: using XGBoost's built-in TreeSHAP (pred_contribs=True).")

X, y, meta = load_compas()
B = binarize(X)
Xtr, Xte, ytr, yte, Btr, Bte = train_test_split(X, y, B, test_size=0.3, stratify=y, random_state=1)
people = np.random.default_rng(0).choice(len(yte), size=min(args.max_people, len(yte)), replace=False)
Xte, Bte, yte = Xte.iloc[people], Bte.iloc[people], yte[people]


def boot(s):
    return np.random.default_rng(s).choice(len(ytr), size=len(ytr), replace=True)


def xgb_run(s):
    idx = boot(s)
    m = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, eval_metric="logloss", random_state=s)
    m.fit(Xtr.iloc[idx], ytr[idx])
    if HAVE_SHAP:
        A = np.asarray(shap.TreeExplainer(m).shap_values(Xte))
    else:
        A = m.get_booster().predict(DMatrix(Xte), pred_contribs=True)[:, :-1]  # drop bias column
    return m.predict(Xte), A


def l1_run(s):
    idx = boot(s)
    m = LogisticRegression(penalty="l1", C=0.05, solver="liblinear").fit(Btr.iloc[idx], ytr[idx])
    return m.predict(Bte), Bte.values * m.coef_[0]  # exact per-person contributions


def tree_run(s):
    idx = boot(s)
    m = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=s).fit(Btr.iloc[idx], ytr[idx])
    path = m.decision_path(Bte).toarray().astype(bool)  # people x nodes
    feat = m.tree_.feature
    used = np.zeros((len(Bte), Bte.shape[1]), dtype=bool)
    for node in np.where(feat >= 0)[0]:
        used[path[:, node], feat[node]] = True
    return m.predict(Bte), used


def topk(A, k):
    """Per-person set of the k features with largest |attribution| (zeros excluded)."""
    order = np.argsort(-np.abs(A), axis=1)[:, :k]
    return [frozenset(j for j in row if A[i, j] != 0) for i, row in enumerate(order)]


def jacc(a, b):
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def stability(runs, is_set=False):
    preds = np.array([r[0] for r in runs])
    n = preds.shape[1]
    res = {"mean_acc": float((preds == yte).mean()), "flip_rate_any": float((preds != preds[0]).any(axis=0).mean())}
    pairs = [(i, j) for i in range(len(runs)) for j in range(i + 1, len(runs))]
    same = np.array([preds[i] == preds[j] for i, j in pairs])  # pairs x people
    for k in args.k:
        if is_set:  # tree: the decision-path feature set is the explanation; k is ignored
            sets = [[frozenset(np.where(r[1][p])[0]) for p in range(n)] for r in runs]
        else:
            sets = [topk(r[1], k) for r in runs]
        J = np.array([[jacc(sets[i][p], sets[j][p]) for p in range(n)] for i, j in pairs])
        res[f"top{k}_jaccard" if not is_set else "path_jaccard"] = float(J.mean())
        res[f"top{k}_jaccard_noflip" if not is_set else "path_jaccard_noflip"] = float(J[same].mean())
        if is_set:
            break
    if not is_set:
        rho = []
        for i, j in pairs[: min(len(pairs), 60)]:  # Spearman is slower; subsample pairs
            for p in range(n):
                a, b = np.abs(runs[i][1][p]), np.abs(runs[j][1][p])
                if a.std() > 0 and b.std() > 0:
                    rho.append(spearmanr(a, b)[0])
        res["rank_spearman"] = float(np.nanmean(rho)) if rho else np.nan
    return res


seeds = range(args.retrains)
table = {"xgboost+TreeSHAP": stability([xgb_run(s) for s in seeds]),
         "sparse_l1 (coef*x)": stability([l1_run(s) for s in seeds]),
         "tree_d3 (decision path)": stability([tree_run(s) for s in seeds], is_set=True)}
T = pd.DataFrame(table).T
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.width", 200)
print(f"\nExplanation stability across {args.retrains} bootstrap retrains (COMPAS, {len(yte)} test people):")
print(T.to_string())
print("\nRead: higher Jaccard / Spearman = more stable explanation. '_noflip' = only people whose decision did not change,"
      "\ni.e. how often the stated reason changes while the outcome stays the same.")
