"""
Feature_Selection_app.py
========================
Standalone Streamlit app that loads the LSWMD wafer dataset, extracts
the full 34-feature set, runs RF-importance-based feature selection via
top_features.py, and displays the top features table.

Run
---
    streamlit run Feature_Selection_app.py
"""

import os
import tempfile
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from Feature_parameters import get_feature_params
from Feature_Extraction import extract_wafer_features
from top_features import select_top_features

st.set_page_config(page_title="Wafer Feature Selection", layout="centered")

_DEFAULT_PATH = os.path.expanduser("~/Documents/DSCI441/LSWMD.pkl")


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
    df = _load_df(path)
    params = get_feature_params()
    feats = df["waferMap"].apply(lambda wm: extract_wafer_features(wm, params))
    return pd.DataFrame(feats.tolist())


@st.cache_data(show_spinner=False)
def _get_labels(path: str) -> pd.Series:
    return _load_df(path)["failureType"].reset_index(drop=True)


_RANDOM_STATE = 42


@st.cache_data(show_spinner=False)
def _run_selection(path: str, n_top: int, n_folds: int):
    """Returns (importance_table, proba_df).

    Feature importances come from an RF fit on the full dataset.
    Predicted probabilities are out-of-fold via StratifiedKFold cross_val_predict.
    """
    features_df = _extract_features(path)
    y           = _get_labels(path)

    X     = features_df.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    y_arr = np.array(y).astype(str)

    # Fit on full data for stable feature importance
    rf = RandomForestClassifier(
        n_estimators=3000, max_features="log2", class_weight="balanced", n_jobs=-1, random_state=_RANDOM_STATE,
    )
    rf.fit(X, y_arr)
    importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_names  = importance.iloc[:n_top].index.tolist()

    table = pd.DataFrame({
        "Rank":       range(1, n_top + 1),
        "Feature":    top_names,
        "Importance": importance[top_names].values.round(5),
    })

    # Out-of-fold predicted probabilities via K-fold cross_val_predict
    # Use only the selected top features so probabilities reflect the chosen subset
    X_top = X[top_names]
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=_RANDOM_STATE)
    rf_cv = RandomForestClassifier(
        n_estimators=3000, max_features="log2", class_weight="balanced", n_jobs=-1, random_state=_RANDOM_STATE,
    )
    proba   = cross_val_predict(rf_cv, X_top, y_arr, cv=cv, method="predict_proba", n_jobs=-1)
    preds   = cross_val_predict(rf_cv, X_top, y_arr, cv=cv, method="predict",       n_jobs=-1)
    classes = list(rf.classes_)
    proba_df = pd.DataFrame(proba, columns=[f"prob_{c}" for c in classes])
    proba_df.insert(0, "sample_idx", np.arange(len(y_arr)))
    proba_df.insert(1, "true_label", y_arr)
    proba_df.insert(2, "pred_label", preds)

    return table, proba_df


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Settings")

uploaded = st.sidebar.file_uploader("Upload dataset (.pkl)", type=["pkl"])
if uploaded is not None:
    # Write uploaded bytes to a stable temp file (cached per upload by name+size)
    upload_key = f"_upload_path_{uploaded.name}_{uploaded.size}"
    if upload_key not in st.session_state:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
        tmp.write(uploaded.read())
        tmp.flush()
        tmp.close()
        st.session_state[upload_key] = tmp.name
    dataset_path = st.session_state[upload_key]
else:
    dataset_path = st.sidebar.text_input("Or enter dataset path (.pkl)", value=_DEFAULT_PATH)
    if not os.path.isfile(dataset_path):
        st.error(f"File not found: `{dataset_path}`")
        st.stop()

n_top       = st.sidebar.slider("Number of top features", min_value=1, max_value=34, value=10)
n_folds     = st.sidebar.slider("K-fold splits (CV)", min_value=2, max_value=10, value=5)

# ── Main ──────────────────────────────────────────────────────────────────────
st.title("Wafer Feature Selection")

with st.spinner("Loading data and extracting features…"):
    _ = _extract_features(dataset_path)   # warm cache

with st.spinner(f"Selecting top {n_top} features + running {n_folds}-fold CV…"):
    top_table, proba_df = _run_selection(dataset_path, n_top, int(n_folds))

st.subheader(f"Top {n_top} Features by RF Importance")
st.dataframe(top_table, use_container_width=True, hide_index=True)

st.subheader(f"Predicted Probabilities ({n_folds}-fold out-of-fold CV)")
st.caption(
    f"{len(proba_df)} samples · {len(proba_df.columns) - 3} classes · "
    "columns: sample_idx, true_label, pred_label, prob_<class>…"
)
st.dataframe(proba_df.head(50), use_container_width=True, hide_index=True)

proba_csv = proba_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇  Download probability CSV",
    data=proba_csv,
    file_name=f"RF_{n_top}features_probabilities.csv",
    mime="text/csv",
)
