"""
feature_hyperparameter_tuning.py
==================================
Unified Streamlit app — hyperparameter search across 10 models on wafer defect data.

Models
------
  RF        RandomForest
  LR        Logistic Regression
  SVM       SVC (kernel)
  LDA       Linear Discriminant Analysis
  QDA       Quadratic Discriminant Analysis
  XGBoost   XGBClassifier
  PCA+LR    PCA → Logistic Regression
  PCA+RF    PCA → Random Forest
  PCA+SVM   PCA → SVC
  PCA+LDA   PCA → LDA
  PCA+QDA   PCA → QDA

Pipeline per model
------------------
1. Load LSWMD dataset  (same pre-processing as Milestone1.py)
2. Extract 34 features via Feature_Extraction.py
3. Select top-N features by PCA variance-weighted importance
4. Sample 200 random combos from each model's grid
5. Evaluate via 80/20 hold-out or Stratified K-Fold CV
6. Metrics: balanced_acc, accuracy, precision/recall/F1 (macro + per-class),
            brier_score, rmse_proba, E_train, E_test, E_train-E_test
7. Auto-download CSV per model  (e.g.  SVM_200_features.csv)
   + manual fallback button

Run
---
    streamlit run feature_hyperparameter_tuning.py
"""

import base64
import itertools
import os
import random
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC, LinearSVC

try:
    from xgboost import XGBClassifier
    _XGB_OK = True
except ImportError:
    _XGB_OK = False

try:
    from imblearn.over_sampling import SMOTE
    _SMOTE_OK = True
except ImportError:
    _SMOTE_OK = False

from Feature_Extraction import extract_wafer_features
from Feature_parameters import get_feature_params
from top_features import select_top_features, select_top_features_pca_lda

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_PATH = os.path.expanduser("~/Documents/DSCI441/LSWMD.pkl")

_CLASS_ORDER = [
    "Scratch", "Loc", "Near-full", "Edge-Ring",
    "Center",  "Donut", "Edge-Loc", "Random",
]

_SVC_MAX_ROWS = 3_000   # SVC is O(n²–n³); cap training rows for speed
_N_SAMPLE     = 200     # combos to evaluate per model (sidebar-adjustable)

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------
_GRIDS = {
    "RF": {
        "n_estimators":  [100, 300, 500],
        "max_features":  ["sqrt", "log2"],
        "max_depth":     [None, 5, 10, 20],
        "min_samples_split": [2, 5],
        "class_weight":  ["balanced", None],
    },
    "LR": {
        "C":             [0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
        "penalty":       ["l2"],
        "solver":        ["lbfgs", "saga"],
        "max_iter":      [500, 1000],
        "class_weight":  ["balanced", None],
    },
    "SVM": {
        "kernel":        ["rbf", "poly", "linear"],
        "C":             [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0],
        "gamma":         ["scale", "auto", 0.001, 0.01, 0.05, 0.10],
        "class_weight":  ["balanced", None],
        "degree":        [2, 3],
    },
    "LDA": {
        "solver":        ["lsqr", "eigen"],
        "shrinkage":     [None, "auto", 0.1, 0.3, 0.5, 0.7, 0.9],
        "tol":           [1e-4, 1e-3, 1e-2],
    },
    "QDA": {
        "reg_param":     [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        "tol":           [1e-4, 1e-3, 1e-2],
    },
    "XGBoost": {
        "n_estimators":  [100, 200, 300],
        "max_depth":     [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample":     [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
        "scale_pos_weight": [1, 3],
    },
    "PCA+LR": {
        "n_components":  [2, 3, 4, 5, 6, 7, 8, 9],
        "C":             [0.01, 0.1, 1.0, 5.0, 10.0, 50.0],
        "solver":        ["lbfgs", "saga"],
        "max_iter":      [500, 1000],
        "class_weight":  ["balanced", None],
    },
    "PCA+RF": {
        "n_components":  [2, 3, 4, 5, 6, 7, 8, 9],
        "n_estimators":  [100, 300, 500],
        "max_features":  ["sqrt", "log2"],
        "max_depth":     [None, 5, 10],
        "class_weight":  ["balanced", None],
    },
    "PCA+SVM": {
        "n_components":  [2, 3, 4, 5, 6, 7, 8, 9],
        "kernel":        ["rbf", "poly", "linear"],
        "C":             [0.1, 0.5, 1.0, 5.0, 10.0, 25.0, 50.0, 100.0],
        "gamma":         ["scale", "auto", 0.001, 0.01, 0.05, 0.10],
        "class_weight":  ["balanced", None],
        "degree":        [2, 3],
    },
    "PCA+LDA": {
        "n_components":  [2, 3, 4, 5, 6, 7, 8, 9],
        "solver":        ["lsqr", "eigen"],
        "shrinkage":     [None, "auto", 0.1, 0.3, 0.5, 0.7, 0.9],
        "tol":           [1e-4, 1e-3, 1e-2],
    },
    "PCA+QDA": {
        "n_components":  [2, 3, 4, 5, 6, 7, 8, 9],
        "reg_param":     [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0],
        "tol":           [1e-4, 1e-3, 1e-2],
    },
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Unified Hyperparameter Tuning",
    page_icon="🔬",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Cached data helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_df(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df["failureType"] = df["failureType"].apply(
        lambda x: np.array(x).flatten()[0] if np.array(x).size > 0 else None
    )
    df = df.dropna(subset=["failureType"])
    df["failureType"] = df["failureType"].astype(str)
    df = df[df["failureType"] != "none"]
    df = df.dropna(subset=["dieSize"])
    df["dieSize"] = df["dieSize"].astype(int)
    df = df[df["dieSize"] < 2500].copy()
    return df


@st.cache_data(show_spinner=False)
def _extract_features(path: str) -> pd.DataFrame:
    df     = _load_df(path)
    params = get_feature_params()
    feats  = df["waferMap"].apply(lambda wm: extract_wafer_features(wm, params))
    return pd.DataFrame(feats.tolist())


@st.cache_data(show_spinner=False)
def _get_labels(path: str) -> np.ndarray:
    return np.array(
        _load_df(path)["failureType"].reset_index(drop=True)
    ).astype(str)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _apply_smote(X_tr: np.ndarray, y_tr: np.ndarray, rng: int) -> tuple:
    if not _SMOTE_OK:
        return X_tr, y_tr
    try:
        y_s    = pd.Series(y_tr, dtype=str)
        counts = y_s.value_counts()
        upsample = {
            cls: int(counts.max())
            for cls, n in counts.items()
            if n < int(counts.max())
        }
        if upsample:
            k  = max(1, min(5, int(counts.min()) - 1))
            sm = SMOTE(sampling_strategy=upsample, k_neighbors=k, random_state=rng)
            X_tr, y_tr = sm.fit_resample(X_tr, y_s)
            y_tr = np.array(y_tr)
    except Exception as exc:
        st.warning(f"SMOTE failed ({exc}) — original training set used.")
    return X_tr, y_tr


def _subsample(X: np.ndarray, y: np.ndarray, cap: int, rng: int) -> tuple:
    if len(y) <= cap:
        return X, y
    _, keep = train_test_split(
        np.arange(len(y)), test_size=cap / len(y), stratify=y, random_state=rng
    )
    return X[keep], y[keep]


def _scale_pca(X_tr, X_te, n_comp, rng):
    sc      = StandardScaler()
    Xtr_s   = sc.fit_transform(X_tr)
    Xte_s   = sc.transform(X_te)
    n       = min(n_comp, Xtr_s.shape[1], Xtr_s.shape[0] - 1)
    pca     = PCA(n_components=n, random_state=rng)
    Xtr_p   = pca.fit_transform(Xtr_s)
    Xte_p   = pca.transform(Xte_s)
    return Xtr_p, Xte_p


def _get_proba(clf, X):
    """Return probability matrix; works for classifiers with or without predict_proba."""
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)
    return None


def _brier_rmse(clf, X_te, y_te):
    y_prob = _get_proba(clf, X_te)
    if y_prob is None:
        return float("nan"), float("nan")
    classes = clf.classes_
    y_bin   = label_binarize(y_te, classes=list(classes))
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    brier = float(np.mean([
        brier_score_loss(y_bin[:, k], y_prob[:, k])
        for k in range(len(classes))
    ]))
    return brier, float(np.sqrt(brier))


def _compute_metrics(params, clf, X_te, y_te, X_tr=None, y_tr=None) -> dict:
    classes = clf.classes_
    cls_ord = (
        [c for c in _CLASS_ORDER if c in classes]
        + [c for c in classes if c not in _CLASS_ORDER]
    )
    y_pred   = clf.predict(X_te)
    ba_te    = balanced_accuracy_score(y_te, y_pred)
    acc_te   = accuracy_score(y_te, y_pred)
    prec_mac = precision_score(y_te, y_pred, average="macro", zero_division=0)
    rec_mac  = recall_score(y_te,  y_pred, average="macro", zero_division=0)
    f1_mac   = f1_score(y_te,      y_pred, average="macro", zero_division=0)

    prec_cls = precision_score(y_te, y_pred, labels=cls_ord, average=None, zero_division=0)
    rec_cls  = recall_score(y_te,  y_pred, labels=cls_ord, average=None, zero_division=0)
    f1_cls   = f1_score(y_te,      y_pred, labels=cls_ord, average=None, zero_division=0)

    brier, rmse_p = _brier_rmse(clf, X_te, y_te)

    if X_tr is not None and y_tr is not None:
        ba_tr   = balanced_accuracy_score(y_tr, clf.predict(X_tr))
        e_train = round(1.0 - ba_tr, 4)
    else:
        ba_tr   = float("nan")
        e_train = float("nan")

    e_test = round(1.0 - ba_te, 4)
    e_gap  = round(e_train - e_test, 4) if not np.isnan(e_train) else float("nan")

    row = {
        **params,
        "balanced_acc":       round(ba_te,    4),
        "train_balanced_acc": round(ba_tr,    4) if not np.isnan(ba_tr) else float("nan"),
        "accuracy":           round(acc_te,   4),
        "precision_macro":    round(prec_mac, 4),
        "recall_macro":       round(rec_mac,  4),
        "f1_macro":           round(f1_mac,   4),
        "brier_score":        round(brier,    4) if not np.isnan(brier) else float("nan"),
        "rmse_proba":         round(rmse_p,   4) if not np.isnan(rmse_p) else float("nan"),
        "E_train":            e_train,
        "E_test":             e_test,
        "E_train-E_test":     e_gap,
    }
    for cls, p, r, f in zip(cls_ord, prec_cls, rec_cls, f1_cls):
        row[f"precision_{cls}"] = round(float(p), 4)
        row[f"recall_{cls}"]    = round(float(r), 4)
        row[f"f1_{cls}"]        = round(float(f), 4)
    return row


def _avg_fold_rows(params, fold_rows) -> dict:
    if not fold_rows:
        return {}
    param_keys = set(params.keys())
    avg = dict(params)
    for k in fold_rows[0].keys():
        if k in param_keys:
            continue
        vals = [
            float(r[k])
            for r in fold_rows
            if r.get(k) is not None and not np.isnan(float(r[k]))
        ]
        avg[k] = round(float(np.mean(vals)), 4) if vals else float("nan")
    return avg


def _sample_combos(grid: dict, n: int, rng_seed: int) -> list:
    keys = list(grid.keys())
    seen, all_combos = set(), []
    for combo in itertools.product(*grid.values()):
        sig = tuple(str(v) for v in combo)
        if sig not in seen:
            seen.add(sig)
            all_combos.append(dict(zip(keys, combo)))
    rng = random.Random(rng_seed)
    return rng.sample(all_combos, min(n, len(all_combos)))


def _select_features_for_model(
    model_name: str,
    features_df: pd.DataFrame,
    y: np.ndarray,
    n_top: int,
    rng: int,
) -> tuple:
    """
    Select top-N features using the model's own importance signal.

    Dispatches:
      RF              → RF mean-decrease-impurity importance
      LR              → |LR coefficients| averaged across classes
      SVM             → |LinearSVC coefficients| averaged across classes
      LDA             → |LDA scalings| averaged across LDA components
      QDA             → PCA variance-weighted importance (QDA has no linear weights)
      XGBoost         → XGB feature_importances_
      PCA+*           → PCA variance-weighted importance (features feed into PCA)

    Returns (top_df, top_names, importance_series, method_label)
    """
    X     = features_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    cols  = list(X.columns)
    y_arr = np.array(y).astype(str)

    X_tr, _, y_tr, _ = train_test_split(
        X.values, y_arr, test_size=0.2, stratify=y_arr, random_state=rng
    )
    sc      = StandardScaler()
    X_tr_s  = sc.fit_transform(X_tr)

    importance = None
    method_label = ""

    if model_name in ("PCA+LR", "PCA+RF", "PCA+SVM", "PCA+LDA", "PCA+QDA", "QDA"):
        # PCA variance-weighted importance
        pca = PCA(random_state=rng)
        pca.fit(X_tr_s)
        pve  = pca.explained_variance_ratio_
        V    = pca.components_           # shape (n_components, n_features)
        importance = np.array([
            float(np.sum(pve * V[:, j] ** 2)) for j in range(len(cols))
        ])
        method_label = "PCA variance-weighted importance"

    elif model_name == "RF":
        pilot = RandomForestClassifier(
            n_estimators=200, max_features="sqrt",
            class_weight="balanced", n_jobs=-1, random_state=rng
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pilot.fit(X_tr_s, y_tr)
        importance = pilot.feature_importances_
        method_label = "RF mean-decrease impurity"

    elif model_name == "LR":
        pilot = LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=500,
            class_weight="balanced", random_state=rng
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pilot.fit(X_tr_s, y_tr)
        importance = np.abs(pilot.coef_).mean(axis=0)
        method_label = "LR |coefficient| mean across classes"

    elif model_name == "SVM":
        pilot = LinearSVC(
            C=1.0, class_weight="balanced",
            max_iter=2000, random_state=rng
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pilot.fit(X_tr_s, y_tr)
        importance = np.abs(pilot.coef_).mean(axis=0)
        method_label = "LinearSVC |coefficient| mean across classes"

    elif model_name == "LDA":
        pilot = LinearDiscriminantAnalysis()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pilot.fit(X_tr_s, y_tr)
        importance = np.abs(pilot.scalings_).mean(axis=1)
        method_label = "LDA |scalings| mean across LD components"

    elif model_name == "XGBoost":
        if _XGB_OK:
            pilot = XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=rng, verbosity=0,
            )
            # XGBoost needs integer labels
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_enc = le.fit_transform(y_tr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pilot.fit(X_tr_s, y_enc)
            importance = pilot.feature_importances_
            method_label = "XGBoost feature_importances_"
        else:
            # fallback
            pca = PCA(random_state=rng)
            pca.fit(X_tr_s)
            pve = pca.explained_variance_ratio_
            V   = pca.components_
            importance = np.array([
                float(np.sum(pve * V[:, j] ** 2)) for j in range(len(cols))
            ])
            method_label = "PCA variance-weighted importance (XGBoost fallback)"

    else:
        # catch-all fallback
        pca = PCA(random_state=rng)
        pca.fit(X_tr_s)
        pve = pca.explained_variance_ratio_
        V   = pca.components_
        importance = np.array([
            float(np.sum(pve * V[:, j] ** 2)) for j in range(len(cols))
        ])
        method_label = "PCA variance-weighted importance"

    importance_series = pd.Series(importance, index=cols).sort_values(ascending=False)
    top_names = importance_series.iloc[:n_top].index.tolist()
    top_df    = X[top_names].copy()
    return top_df, top_names, importance_series, method_label


def _build_clf(model_name: str, params: dict, rng: int):
    """Instantiate the classifier from model_name and params dict."""
    p = dict(params)
    # Remove PCA key — handled externally
    n_comp = p.pop("n_components", None)

    if model_name == "RF":
        return n_comp, RandomForestClassifier(
            **p, n_jobs=-1, random_state=rng
        )
    elif model_name == "LR":
        penalty = p.get("penalty", "l2")
        solver  = p.get("solver", "lbfgs")
        # saga supports l1/l2; lbfgs supports only l2
        if penalty == "l1" and solver == "lbfgs":
            p["solver"] = "saga"
        return n_comp, LogisticRegression(**p, random_state=rng)
    elif model_name == "SVM":
        return n_comp, SVC(**p, probability=True, random_state=rng)
    elif model_name == "LDA":
        shrink = p.get("shrinkage")
        solver = p.get("solver", "lsqr")
        if shrink is not None and solver not in ("lsqr", "eigen"):
            p["solver"] = "lsqr"
        return n_comp, LinearDiscriminantAnalysis(**p)
    elif model_name == "QDA":
        return n_comp, QuadraticDiscriminantAnalysis(**p)
    elif model_name == "XGBoost":
        if not _XGB_OK:
            return n_comp, None
        return n_comp, XGBClassifier(
            **p,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=rng,
            verbosity=0,
        )
    elif model_name == "PCA+LR":
        penalty = p.get("penalty", "l2")
        solver  = p.get("solver", "lbfgs")
        if penalty == "l1" and solver == "lbfgs":
            p["solver"] = "saga"
        return n_comp, LogisticRegression(**p, random_state=rng)
    elif model_name == "PCA+RF":
        return n_comp, RandomForestClassifier(**p, n_jobs=-1, random_state=rng)
    elif model_name == "PCA+SVM":
        return n_comp, SVC(**p, probability=True, random_state=rng)
    elif model_name == "PCA+LDA":
        shrink = p.get("shrinkage")
        solver = p.get("solver", "lsqr")
        if shrink is not None and solver not in ("lsqr", "eigen"):
            p["solver"] = "lsqr"
        return n_comp, LinearDiscriminantAnalysis(**p)
    elif model_name == "PCA+QDA":
        return n_comp, QuadraticDiscriminantAnalysis(**p)
    return n_comp, None


def _uses_pca(model_name: str) -> bool:
    return model_name.startswith("PCA+")


def _needs_svc_cap(model_name: str) -> bool:
    return model_name in ("SVM", "PCA+SVM")


def _auto_download_js(b64: str, filename: str) -> str:
    return f"""
    <script>
      (function() {{
        var a  = document.createElement('a');
        a.href = 'data:text/csv;base64,{b64}';
        a.download = '{filename}';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
      }})();
    </script>
    """


# ---------------------------------------------------------------------------
# Run search for ONE model
# ---------------------------------------------------------------------------
def _run_model_search(
    model_name,
    X_top, y,
    sampled_combos,
    use_cv, n_folds,
    use_smote, svc_cap, rng,
) -> pd.DataFrame:
    records = []
    total   = len(sampled_combos)
    t0      = time.time()
    pbar    = st.progress(0)
    status  = st.empty()

    needs_cap = _needs_svc_cap(model_name)
    needs_pca = _uses_pca(model_name)

    skf   = StratifiedKFold(n_splits=int(n_folds), shuffle=True, random_state=rng)
    folds = list(skf.split(X_top, y))

    for i, params in enumerate(sampled_combos, start=1):
        try:
            fold_rows = []
            for tr_idx, te_idx in folds:
                X_tr_f, X_te_f = X_top[tr_idx], X_top[te_idx]
                y_tr_f, y_te_f = y[tr_idx], y[te_idx]
                if use_smote:
                    X_tr_f, y_tr_f = _apply_smote(X_tr_f, y_tr_f, rng)
                X_tr_gs, y_tr_gs = (
                    _subsample(X_tr_f, y_tr_f, svc_cap, rng)
                    if needs_cap else (X_tr_f, y_tr_f)
                )
                n_comp, clf = _build_clf(model_name, params, rng)
                if clf is None:
                    continue
                if needs_pca:
                    Xtr_p, Xte_p = _scale_pca(X_tr_gs, X_te_f, n_comp, rng)
                else:
                    sc      = StandardScaler()
                    Xtr_p   = sc.fit_transform(X_tr_gs)
                    Xte_p   = sc.transform(X_te_f)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf.fit(Xtr_p, y_tr_gs)
                fold_rows.append(
                    _compute_metrics(params, clf, Xte_p, y_te_f, Xtr_p, y_tr_gs)
                )
            if fold_rows:
                records.append(_avg_fold_rows(params, fold_rows))
        except Exception:
            pass

        pbar.progress(i / total)
        status.text(
            f"{model_name}  combo {i}/{total}  |  elapsed {time.time()-t0:.1f}s"
        )

    pbar.empty()
    status.empty()

    if not records:
        return pd.DataFrame()
    return (
        pd.DataFrame(records)
        .sort_values("balanced_acc", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.title("🔬 Unified Hyperparameter Tuning — All Models")
st.markdown(
    "Runs a random grid search for each selected model. "
    "Features are selected **per model** using that model's own importance signal. "
    "PCA-prefixed models apply StandardScaler + PCA before the classifier. "
    "A CSV is auto-downloaded for every model on completion."
)

st.sidebar.header("⚙️ Settings")

# ── 1. Dataset ────────────────────────────────────────────────────────────────
st.sidebar.subheader("📂 Dataset")
uploaded = st.sidebar.file_uploader("Upload dataset (.pkl)", type=["pkl"])
if uploaded is not None:
    upload_key = f"_upload_{uploaded.name}_{uploaded.size}"
    if upload_key not in st.session_state:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        tmp.write(uploaded.read())
        tmp.flush(); tmp.close()
        st.session_state[upload_key] = tmp.name
    dataset_path = st.session_state[upload_key]
else:
    dataset_path = st.sidebar.text_input(
        "Or enter dataset path (.pkl)", value=_DEFAULT_PATH
    )
    if not os.path.isfile(dataset_path):
        st.error(f"File not found: `{dataset_path}`")
        st.stop()

# ── 2. Model selection ────────────────────────────────────────────────────────
st.sidebar.subheader("🤖 Models")
all_models = [
    "RF", "LR", "SVM", "LDA", "QDA",
    "XGBoost",
    "PCA+LR", "PCA+RF", "PCA+SVM", "PCA+LDA", "PCA+QDA",
]
if not _XGB_OK:
    all_models.remove("XGBoost")
    st.sidebar.warning("XGBoost not installed — disabled.")

selected_models = st.sidebar.multiselect(
    "Models to run",
    options=all_models,
    default=all_models,
    help=(
        "RF=Random Forest, LR=Logistic Regression, SVM=Support Vector Machine, "
        "LDA=Linear Discriminant Analysis, QDA=Quadratic Discriminant Analysis. "
        "PCA+ prefix applies dimensionality reduction first."
    ),
)

# ── 3. Feature selection ──────────────────────────────────────────────────────
st.sidebar.subheader("🔍 Feature Selection")
n_top = st.sidebar.slider(
    "Top N features per model", min_value=1, max_value=34, value=10,
    help=(
        "Each model selects features using its own signal: "
        "RF→impurity, LR→|coef|, SVM→LinearSVC|coef|, "
        "LDA→|scalings|, XGBoost→XGB importance, PCA+*→PCA variance."
    ),
)

# ── 4. Evaluation mode ────────────────────────────────────────────────────────
st.sidebar.subheader("📊 Evaluation")
n_folds   = st.sidebar.slider(
    "CV folds", 3, 10, 5,
    help="Number of stratified folds for Stratified K-Fold CV.",
)
use_cv    = True  # always K-Fold

# ── 5. Grid search ────────────────────────────────────────────────────────────
st.sidebar.subheader("⚡ Grid Search")
n_sample = st.sidebar.slider(
    "Combos per model", min_value=50, max_value=500, value=_N_SAMPLE, step=50,
    help="Number of random hyperparameter combinations evaluated per model.",
)
use_smote = st.sidebar.checkbox(
    "SMOTE oversampling (train split only)",
    value=False,
    disabled=not _SMOTE_OK,
    help=(
        "Oversample minority classes in the training split using SMOTE. "
        "Requires imbalanced-learn (`pip install imbalanced-learn`)."
        + (" ✅ Available." if _SMOTE_OK else " ❌ Not installed.")
    ),
)
svc_cap = _SVC_MAX_ROWS  # fixed at 3 000; SVC is O(n²–n³)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("▶  Run All Selected Models", type="primary")

# ---------------------------------------------------------------------------
# Step 0 – Load & extract
# ---------------------------------------------------------------------------
with st.spinner("Loading dataset and extracting features…"):
    features_df = _extract_features(dataset_path)
    y           = _get_labels(dataset_path)

n_total, n_feat = features_df.shape
n_classes       = len(np.unique(y))

c1, c2, c3 = st.columns(3)
c1.metric("Total samples",  f"{n_total:,}")
c2.metric("Total features", n_feat)
c3.metric("Defect classes", n_classes)

# ---------------------------------------------------------------------------
# Step 1 – Guard
# ---------------------------------------------------------------------------
st.info(
    f"Top **{n_top}** features will be selected **per model** using each model's "
    "own importance signal (RF → RF impurity, LR → |coef|, SVM → LinearSVC |coef|, "
    "LDA → |scalings|, XGBoost → XGB importance, PCA+* / QDA → PCA variance-weighted). "
    "All selections fit on the training split only — no data leakage."
)

if not run_btn:
    st.info("Configure settings in the sidebar, then click **▶ Run All Selected Models**.")
    st.stop()

if not selected_models:
    st.warning("No models selected.")
    st.stop()

rng = 42  # fixed random state

# ---------------------------------------------------------------------------
# Step 2 – Run each model
# ---------------------------------------------------------------------------
st.subheader("Step 2 — Grid Search Results")

_PRIMARY_COLS = [
    "balanced_acc", "train_balanced_acc",
    "accuracy", "precision_macro", "recall_macro", "f1_macro",
    "brier_score", "rmse_proba", "E_train", "E_test", "E_train-E_test",
]

all_results   = {}   # model_name → DataFrame
download_html = []   # accumulate JS snippets

for model_name in selected_models:
    st.markdown(f"---\n### 🤖 {model_name}")

    if model_name == "XGBoost" and not _XGB_OK:
        st.warning("XGBoost not installed — skipping.")
        continue

    # ── Model-specific feature selection ──────────────────────────────────
    with st.spinner(f"[{model_name}] Selecting top {n_top} features…"):
        top_df, top_names, importance_series, method_label = _select_features_for_model(
            model_name, features_df, y, n_top, rng
        )

    feat_table = pd.DataFrame({
        "Rank":       range(1, n_top + 1),
        "Feature":    top_names,
        "Importance": importance_series[top_names].round(5).values,
    })
    st.markdown(f"**Feature selection method:** {method_label}")
    st.dataframe(feat_table, use_container_width=True, hide_index=True)

    fig_feat = px.bar(
        feat_table, x="Feature", y="Importance",
        title=f"{model_name} — Top {n_top} Features ({method_label})",
        color="Importance", color_continuous_scale="Blues",
    )
    fig_feat.update_layout(xaxis_tickangle=-35, coloraxis_showscale=False)
    st.plotly_chart(fig_feat, use_container_width=True)

    X_top = top_df.fillna(0.0).replace([np.inf, -np.inf], 0.0).values

    # Cap n_components for PCA models to never exceed n_top
    grid = dict(_GRIDS[model_name])
    if "n_components" in grid:
        grid["n_components"] = [c for c in grid["n_components"] if c <= len(top_names)]

    sampled = _sample_combos(grid, n_sample, rng)
    n_unique = len(list(itertools.islice(
        (dict(zip(grid.keys(), c)) for c in itertools.product(*grid.values())), 10_000
    )))
    st.info(
        f"Grid combos: **{len(sampled)}** sampled  |  "
        f"Mode: **Stratified K-Fold CV ({n_folds} folds)**  |  SMOTE: **{use_smote}**"
        + (f"  |  SVC cap: **{_SVC_MAX_ROWS:,}** rows" if _needs_svc_cap(model_name) else "")
    )

    results_df = _run_model_search(
        model_name, X_top, y,
        sampled, use_cv, int(n_folds),
        use_smote, int(svc_cap), rng,
    )

    if results_df.empty:
        st.error(f"No results for {model_name}. Check dataset and parameters.")
        continue

    all_results[model_name] = results_df

    # Display columns
    param_cols   = [k for k in _GRIDS[model_name].keys()]
    display_cols = (
        param_cols
        + [c for c in _PRIMARY_COLS if c in results_df.columns]
    )

    st.markdown("#### 🏆 Top-5 Configurations")
    st.dataframe(
        results_df[display_cols].head(5).style
        .highlight_max(subset=["balanced_acc", "f1_macro"], color="#d4f1d4")
        .highlight_min(subset=["brier_score",  "E_test"],   color="#d4f1d4"),
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(f"Full results — {model_name} (top 100 rows)"):
        st.dataframe(results_df[display_cols].head(100), use_container_width=True)

    # Per-class metrics for best config
    best = results_df.iloc[0]
    per_class_data = [
        {
            "Class":     cls,
            "Precision": best.get(f"precision_{cls}", float("nan")),
            "Recall":    best.get(f"recall_{cls}",    float("nan")),
            "F1":        best.get(f"f1_{cls}",        float("nan")),
        }
        for cls in _CLASS_ORDER
        if f"recall_{cls}" in best.index
    ]
    if per_class_data:
        st.markdown(
            f"**Per-class metrics — best config:** "
            + ",  ".join(f"{k}={best[k]}" for k in param_cols if k in best.index)
        )
        st.dataframe(pd.DataFrame(per_class_data), use_container_width=True, hide_index=True)

    # Bias-Variance scatter
    if {"E_train", "E_test"}.issubset(results_df.columns):
        bv   = results_df.dropna(subset=["E_train", "E_test"])
        color_col = "kernel" if "kernel" in bv.columns else (param_cols[1] if len(param_cols) > 1 else param_cols[0])
        fig_bv = px.scatter(
            bv, x="E_train", y="E_test", color=color_col,
            hover_data=[c for c in param_cols if c in bv.columns] + ["balanced_acc"],
            title=f"{model_name} — Bias-Variance: E_train vs E_test",
            labels={"E_train": "E_train (1−Bal.Acc)", "E_test": "E_test (1−Bal.Acc)"},
        )
        mx = float(bv[["E_train", "E_test"]].max().max()) * 1.05
        fig_bv.add_shape(type="line", x0=0, y0=0, x1=mx, y1=mx,
                         line=dict(dash="dash", color="gray", width=1))
        st.plotly_chart(fig_bv, use_container_width=True)

    # CSV filename:  e.g.  SVM_200_features.csv
    save_dir   = os.path.dirname(dataset_path) if os.path.isabs(dataset_path) else os.getcwd()
    csv_name   = f"{model_name.replace('+', '_')}_{n_sample}_features.csv"
    csv_path   = os.path.join(save_dir, csv_name)
    results_df.to_csv(csv_path, index=False)          # ← saved to disk immediately
    csv_bytes  = results_df.to_csv(index=False).encode("utf-8")
    b64        = base64.b64encode(csv_bytes).decode()
    st.success(f"💾 Saved **{csv_name}** → `{csv_path}`")

    # Auto-download JS — fire immediately per model with a small delay
    # so browser doesn't block simultaneous downloads.
    model_idx  = selected_models.index(model_name)
    delay_ms   = model_idx * 800   # stagger by 800 ms per model
    st.components.v1.html(
        f"""
        <script>
          setTimeout(function() {{
            var a  = document.createElement('a');
            a.href = 'data:text/csv;base64,{b64}';
            a.download = '{csv_name}';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
          }}, {delay_ms});
        </script>
        """,
        height=0,
    )
    download_html.append(csv_name)   # track names for final summary

    # Manual fallback button
    st.download_button(
        label=f"⬇  Download {csv_name}",
        data=csv_bytes,
        file_name=csv_name,
        mime="text/csv",
        key=f"dl_{model_name}",
    )

    # ── Per-sample probability CSV (best config, K-fold cross_val_predict) ──
    try:
        best_params_dict = {
            k: results_df.iloc[0][k]
            for k in _GRIDS[model_name].keys()
            if k in results_df.columns
        }
        n_comp_best, clf_best = _build_clf(model_name, best_params_dict, rng)
        if clf_best is not None and hasattr(clf_best, "predict_proba"):
            cv_prob = StratifiedKFold(n_splits=int(n_folds), shuffle=True, random_state=rng)
            if _uses_pca(model_name) and n_comp_best:
                from sklearn.pipeline import Pipeline
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("pca",    PCA(n_components=int(n_comp_best), random_state=rng)),
                    ("clf",    clf_best),
                ])
            else:
                from sklearn.pipeline import Pipeline
                pipe = Pipeline([
                    ("scaler", StandardScaler()),
                    ("clf",    clf_best),
                ])
            import warnings as _w
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                proba_oof = cross_val_predict(
                    pipe, X_top, y, cv=cv_prob,
                    method="predict_proba", n_jobs=-1,
                )
                preds_oof = cross_val_predict(
                    pipe, X_top, y, cv=cv_prob,
                    method="predict", n_jobs=-1,
                )
            classes_list = sorted(set(y.tolist()))
            prob_df = pd.DataFrame(proba_oof, columns=[f"prob_{c}" for c in classes_list])
            prob_df.insert(0, "sample_idx", np.arange(len(y)))
            prob_df.insert(1, "true_label",  y)
            prob_df.insert(2, "pred_label",  preds_oof)

            prob_csv_name  = f"{model_name.replace('+', '_')}_{n_sample}_probabilities.csv"
            prob_csv_path  = os.path.join(save_dir, prob_csv_name)
            prob_csv_bytes = prob_df.to_csv(index=False).encode("utf-8")
            prob_df.to_csv(prob_csv_path, index=False)
            st.success(f"💾 Saved **{prob_csv_name}** → `{prob_csv_path}`")
            st.download_button(
                label=f"⬇  Download {prob_csv_name}",
                data=prob_csv_bytes,
                file_name=prob_csv_name,
                mime="text/csv",
                key=f"dl_prob_{model_name}",
            )
            st.caption(
                f"{len(prob_df)} samples · {len(classes_list)} classes · "
                "out-of-fold K-fold probabilities for best hyperparameter config"
            )
    except Exception as _pe:
        st.warning(f"Probability CSV for {model_name} skipped: {_pe}")

    # Summary metrics
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Best Balanced Acc", f"{best['balanced_acc']:.4f}")
    m2.metric("Best F1 Macro",
              f"{best.get('f1_macro', float('nan')):.4f}" if "f1_macro" in best.index else "—")
    m3.metric("Best Brier Score",
              f"{best.get('brier_score', float('nan')):.4f}" if "brier_score" in best.index else "—")
    m4.metric("E_test (best)",
              f"{best.get('E_test', float('nan')):.4f}" if "E_test" in best.index else "—")
    m5.metric("E_train-E_test",
              f"{best.get('E_train-E_test', float('nan')):.4f}" if "E_train-E_test" in best.index else "—")

# ---------------------------------------------------------------------------
# Step 4 – Cross-model comparison
# ---------------------------------------------------------------------------
if len(all_results) > 1:
    st.markdown("---")
    st.subheader("Step 3 — Cross-Model Comparison")

    summary_rows = []
    for mn, df in all_results.items():
        b = df.iloc[0]
        summary_rows.append({
            "Model":         mn,
            "Balanced Acc":  b.get("balanced_acc",        float("nan")),
            "F1 Macro":      b.get("f1_macro",            float("nan")),
            "Brier Score":   b.get("brier_score",         float("nan")),
            "E_test":        b.get("E_test",              float("nan")),
            "E_train-E_test": b.get("E_train-E_test",    float("nan")),
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("Balanced Acc", ascending=False)

    st.markdown("#### Best config per model")
    st.dataframe(
        summary_df.style
        .highlight_max(subset=["Balanced Acc", "F1 Macro"], color="#d4f1d4")
        .highlight_min(subset=["Brier Score",  "E_test"],   color="#d4f1d4"),
        use_container_width=True,
        hide_index=True,
    )

    fig_cmp = px.bar(
        summary_df,
        x="Model", y="Balanced Acc",
        color="Model",
        title="Best Balanced Accuracy per Model",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    fig_cmp.update_layout(showlegend=False)
    st.plotly_chart(fig_cmp, use_container_width=True)

    fig_f1 = px.bar(
        summary_df,
        x="Model", y="F1 Macro",
        color="Model",
        title="Best F1 Macro per Model",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_f1.update_layout(showlegend=False)
    st.plotly_chart(fig_f1, use_container_width=True)

# ---------------------------------------------------------------------------
# Step 5 – Final summary
# ---------------------------------------------------------------------------
if download_html:
    st.success(
        f"✅ Search complete for **{len(all_results)}** model(s). "
        f"Auto-downloaded: {', '.join(f'`{n}`' for n in download_html)}. "
        "Use the ⬇ buttons above if any file didn't save automatically."
    )
