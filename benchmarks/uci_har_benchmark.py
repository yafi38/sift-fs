# %% [markdown]
# # UCI HAR Benchmark: sift-fs vs. Baselines
#
# Evaluates sift-fs on the UCI Human Activity Recognition dataset (561 features,
# 6 activities, 10 299 samples) against three standard feature-selection baselines:
# Random Forest RFE, Lasso (L1 logistic regression), and Mutual Information.
#
# Downstream classifiers: Random Forest, SVM, KNN.
# Metrics: accuracy and macro-F1 at varying feature budgets.
# Hardware metric: per-sensor activation (accelerometer vs. gyroscope).

# %% Setup — run `pip install` on Kaggle; harmless locally if already installed
# !pip install -q "siftfs @ git+https://github.com/yafi38/sift-fs@main"

# %%
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from siftfs import FeatureSelector

warnings.filterwarnings("ignore", category=FutureWarning)

FIGURES_DIR = Path.cwd() / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PANEL_SIZES = [5, 10, 20, 50, 100, 200, 300]
RANDOM_STATE = 42

# Set False to run only sift-fs and skip the RF-RFE / Lasso / MI baselines
# (useful while developing sift-fs — the baselines are the slow part).
RUN_ALL_METHODS = False

# %% [markdown]
# ## 1. Load UCI HAR Dataset

# %%
DATA_DIR = Path.cwd() / "uci_har"
DATA_DIR.mkdir(exist_ok=True)


def _find_har_root() -> Path:
    """Locate the extracted UCI HAR Dataset directory.

    Order of preference:
      1. A previously extracted copy in ``DATA_DIR``.
      2. The ``/kaggle/input`` directory (Upload Data > added a HAR dataset).
      3. The official UCI ML Repository zip, downloaded and extracted.
    """
    for cand in (DATA_DIR / "UCI HAR Dataset",):
        if cand.exists():
            return cand

    # 2. Kaggle Input Data
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        # Public datasets standardise on this layout under /kaggle/input/...;
        # glob for the folder that actually contains features.txt.
        hits = sorted(kaggle_input.glob("*/UCI HAR Dataset"))
        if hits:
            return hits[0]
        for sub in sorted(kaggle_input.glob("*")):
            if (sub / "features.txt").exists():
                return sub
        # Some uploads nest one level deeper.
        for sub in sorted(kaggle_input.glob("*/*")):
            if (sub / "features.txt").exists():
                return sub

    # 3. Download from UCI.
    if not (DATA_DIR / "uci_har.zip").exists():
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
        print(f"Downloading UCI HAR from {url} ...")
        import urllib.request

        resp = urllib.request.urlopen(url)
        (DATA_DIR / "uci_har.zip").write_bytes(resp.read())
    zip_path = DATA_DIR / "uci_har.zip"
    if not (DATA_DIR / "UCI HAR Dataset").exists():
        import zipfile

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DATA_DIR)
    return DATA_DIR / "UCI HAR Dataset"


base = _find_har_root()
print(f"Using UCI HAR data from: {base}")

feature_names = [
    line.split(None, 1)[1]
    for line in (base / "features.txt").read_text().splitlines()
]

activity_labels = {}
for line in (base / "activity_labels.txt").read_text().splitlines():
    idx, name = line.split()
    activity_labels[int(idx)] = name.lower().replace("_", " ")

X_train = np.loadtxt(base / "train" / "X_train.txt")
y_train = np.loadtxt(base / "train" / "y_train.txt", dtype=int)
X_test = np.loadtxt(base / "test" / "X_test.txt")
y_test = np.loadtxt(base / "test" / "y_test.txt", dtype=int)

X_all = np.vstack([X_train, X_test])
y_all = np.concatenate([y_train, y_test])

print(f"Train: {X_train.shape}  Test: {X_test.shape}")
print(f"Activities: {activity_labels}")
print(f"Features: {len(feature_names)}")

# %% [markdown]
# ## 2. Sensor Grouping (reporting only)
#
# Map each of the 561 features to its physical sensor origin so we can report the
# sensor activation per method. This is *not* passed to sift-fs as a group
# penalty — it mirrors scGIST's UCI benchmark, which applies no pairs/group
# constraint.
# - **Accelerometer** — feature name contains `Acc`
# - **Gyroscope** — feature name contains `Gyro`
# - **Other** — `angle(...)` features (3 total)

# %%
acc_idx = [i for i, n in enumerate(feature_names) if "Acc" in n]
gyro_idx = [i for i, n in enumerate(feature_names) if "Gyro" in n]
other_idx = [i for i in range(len(feature_names)) if i not in acc_idx and i not in gyro_idx]

print(f"Accelerometer features: {len(acc_idx)}")
print(f"Gyroscope features:     {len(gyro_idx)}")
print(f"Other (angle) features: {len(other_idx)}")
assert len(acc_idx) + len(gyro_idx) + len(other_idx) == len(feature_names)

# %% [markdown]
# ## 3. Downstream Classifiers

# %%
def make_classifiers() -> dict:
    return {
        "RF": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "SVM": SVC(kernel="rbf", random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }


def evaluate(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Train each downstream classifier and return {name: {acc, macro_f1}}."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    results = {}
    for clf_name, clf in make_classifiers().items():
        from sklearn.base import clone
        c = clone(clf)
        c.fit(X_tr_s, y_tr)
        y_pred = c.predict(X_te_s)
        results[clf_name] = {
            "acc": accuracy_score(y_te, y_pred),
            "macro_f1": f1_score(y_te, y_pred, average="macro"),
        }
    return results

# %% [markdown]
# ## 4. Feature Selection Methods

# %%


def run_siftfs(
    X_tr: np.ndarray, y_tr: np.ndarray, panel_size: int,
) -> np.ndarray:
    """Run sift-fs and return selected feature indices.

    Mirrors the scGIST UCI benchmark exactly: same hidden topology ``(32, 16)``,
    same ``alpha=1.5``, fixed 200 epochs (no early stopping), no group/pairs
    constraint.
    """
    sel = FeatureSelector(
        panel_size=panel_size,
        l1=0.01,
        hidden_dims=(32, 16),
        alpha=1.5,
        epochs=200,
        early_stop=False,
        seed=RANDOM_STATE,
    )
    sel.fit(X_tr, y_tr)
    return np.array(sel.get_selected_features())


def run_rfe(
    X_tr: np.ndarray, y_tr: np.ndarray, panel_size: int,
) -> np.ndarray:
    """Run Recursive Feature Elimination with Random Forest."""
    estimator = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1)
    rfe = RFE(estimator, n_features_to_select=panel_size, step=max(1, panel_size // 5))
    rfe.fit(X_tr, y_tr)
    return np.where(rfe.support_)[0]


def run_lasso(
    X_tr: np.ndarray, y_tr: np.ndarray, panel_size: int,
) -> np.ndarray:
    """L1 logistic regression — select top features by coefficient magnitude."""
    from sklearn.multiclass import OneVsRestClassifier

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    base = LogisticRegression(
        penalty="l1", solver="liblinear", C=0.1, max_iter=500,
        random_state=RANDOM_STATE,
    )
    lr = OneVsRestClassifier(base)
    lr.fit(X_tr_s, y_tr)
    coefs = np.vstack([est.coef_.ravel() for est in lr.estimators_])
    importance = np.mean(np.abs(coefs), axis=0)
    return np.argsort(importance)[::-1][:panel_size]


def run_mi(
    X_tr: np.ndarray, y_tr: np.ndarray, panel_size: int,
) -> np.ndarray:
    """Mutual Information — select top features by MI score."""
    mi_scores = mutual_info_classif(X_tr, y_tr, random_state=RANDOM_STATE, n_neighbors=5)
    return np.argsort(mi_scores)[::-1][:panel_size]


METHODS = {
    "sift-fs": run_siftfs,
    "RF-RFE": run_rfe,
    "Lasso": run_lasso,
    "MI": run_mi,
}

# Honor the RUN_ALL_METHODS flag: when False, only run sift-fs.
methods_to_run = METHODS if RUN_ALL_METHODS else {"sift-fs": METHODS["sift-fs"]}

# %% [markdown]
# ## 5. Run Benchmark

# %%
results_rows: list[dict] = []
selected_features_cache: dict[tuple[str, int], np.ndarray] = {}

for method_name, method_fn in methods_to_run.items():
    for ps in PANEL_SIZES:
        print(f"  {method_name:8s}  d={ps:3d}  ...", end=" ", flush=True)
        indices = method_fn(X_train, y_train, ps)
        selected_features_cache[(method_name, ps)] = indices

        # Count per-sensor activation
        n_acc = sum(1 for i in indices if i in acc_idx)
        n_gyro = sum(1 for i in indices if i in gyro_idx)
        n_other = sum(1 for i in indices if i in other_idx)

        clf_results = evaluate(X_train[:, indices], y_train, X_test[:, indices], y_test)
        for clf_name, metrics in clf_results.items():
            results_rows.append({
                "method": method_name,
                "panel_size": ps,
                "classifier": clf_name,
                "n_acc": n_acc,
                "n_gyro": n_gyro,
                "n_other": n_other,
                "sensors_on": sum([n_acc > 0, n_gyro > 0]),
                **metrics,
            })
        best_clf = max(clf_results, key=lambda k: clf_results[k]["acc"])
        print(
            f"best={best_clf}  acc={clf_results[best_clf]['acc']:.3f}"
            f"  macro_f1={clf_results[best_clf]['macro_f1']:.3f}"
        )

df = pd.DataFrame(results_rows)
print("\n", df.pivot_table(
    index=["method", "panel_size"], columns="classifier", values="acc",
).round(3))

# %% [markdown]
# ## 6. Pareto Curves — Accuracy vs. Feature Count

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
colors = {"sift-fs": "#e74c3c", "RF-RFE": "#3498db", "Lasso": "#2ecc71", "MI": "#9b59b6"}

for ax, clf_name in zip(axes, ["RF", "SVM", "KNN"]):
    subset = df[df["classifier"] == clf_name]
    for method_name in methods_to_run:
        mdf = subset[subset["method"] == method_name].sort_values("panel_size")
        ax.plot(mdf["panel_size"], mdf["acc"], "o-", label=method_name,
                color=colors[method_name], linewidth=2, markersize=5)
    ax.set_title(f"Downstream: {clf_name}", fontsize=12)
    ax.set_xlabel("Features selected (d)")
    ax.set_ylabel("Accuracy" if clf_name == "RF" else "")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

fig.suptitle("UCI HAR — Accuracy vs. Feature Budget", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "pareto_accuracy.png", dpi=150, bbox_inches="tight")
print(f"Saved: {FIGURES_DIR / 'pareto_accuracy.png'}")

# %% [markdown]
# ## 7. Hardware Metric — Sensor Activation

# %%
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for ax, clf_name in zip(axes, ["RF", "SVM", "KNN"]):
    subset = df[df["classifier"] == clf_name]
    for i, method_name in enumerate(methods_to_run):
        mdf = subset[subset["method"] == method_name].sort_values("panel_size")
        x = np.arange(len(mdf))
        w = 0.18
        offset = i * w - 1.5 * w
        ax.bar(
            x + offset, mdf["n_acc"], w,
            label=f"{method_name} Acc", color=colors[method_name], alpha=0.8,
        )
        ax.bar(
            x + offset, mdf["n_gyro"], w, bottom=mdf["n_acc"],
            label=f"{method_name} Gyro", color=colors[method_name], alpha=0.4,
        )
    ax.set_title(f"Downstream: {clf_name}", fontsize=12)
    ax.set_xlabel("Panel size")
    ax.set_ylabel("Features" if clf_name == "RF" else "")
    ax.set_xticks(range(len(PANEL_SIZES)))
    ax.set_xticklabels([str(p) for p in PANEL_SIZES])
    ax.grid(True, alpha=0.3, axis="y")

fig.suptitle("UCI HAR — Sensor Feature Activation (stacked Acc + Gyro)", fontsize=14, y=1.02)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "sensor_activation.png", dpi=150, bbox_inches="tight")
print(f"Saved: {FIGURES_DIR / 'sensor_activation.png'}")

# %% [markdown]
# ## 8. Summary Table

# %%
# Each (method, panel_size, classifier) combo has exactly one row, so no true
# aggregation is needed — just select the columns and de-duplicate. This avoids
# the pandas 3.x change that dropped the string-kwarg `.agg(new="old")` form.
cols = ["method", "panel_size", "classifier", "acc", "macro_f1", "sensors_on", "n_acc", "n_gyro"]
best_per_method = df[cols].drop_duplicates(subset=["method", "panel_size", "classifier"])
best_per_method = best_per_method.sort_values(["classifier", "panel_size", "method"])
print(best_per_method.to_string(index=False, float_format="{:.3f}".format))

# Save full results to CSV
csv_path = FIGURES_DIR / "uci_har_results.csv"
df.to_csv(csv_path, index=False)
print(f"\nSaved full results to {csv_path}")
