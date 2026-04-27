import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from imblearn.over_sampling import SMOTE
from Feature_parameters import get_feature_params
from Feature_Extraction import extract_wafer_features
from top_features import select_top_features

st.set_page_config(page_title="Stacking Ensemble — SVM Meta-Model", layout="wide")

_DEFAULT_PATH  = os.path.expanduser("~/Documents/DSCI441/LSWMD_labeled.pkl")
_WORK_DIR      = os.path.dirname(os.path.abspath(__file__))
_RANDOM_STATE  = 42

_MODEL_CONFIGS = {
    "LR": {
        "feat_csv": "LR_200_features.csv",
        "prob_csv": "LR_200_probabilities.csv",
        "hp_cols":  ["C", "penalty", "solver", "max_iter", "class_weight"],
    },
    "SVM": {
        "feat_csv": "SVM_200_features.csv",
        "prob_csv": "SVM_200_probabilities.csv",
        "hp_cols":  ["kernel", "C", "gamma", "class_weight", "degree"],
    },
    "LDA": {
        "feat_csv": "LDA_200_features.csv",
        "prob_csv": "LDA_200_probabilities.csv",
        "hp_cols":  ["solver", "shrinkage", "tol"],
    },
    "QDA": {
        "feat_csv": "QDA_200_features.csv",
        "prob_csv": "QDA_200_probabilities.csv",
        "hp_cols":  ["reg_param", "tol"],
    },
    "PCA+LDA": {
        "feat_csv": "PCA_LDA_200_features.csv",
        "prob_csv": "PCA_LDA_200_probabilities.csv",
        "hp_cols":  ["n_components", "solver", "shrinkage", "tol"],
    },
    "PCA+LR": {
        "feat_csv": "PCA_LR_200_features.csv",
        "prob_csv": "PCA_LR_200_probabilities.csv",
        "hp_cols":  ["n_components", "C", "solver", "max_iter", "class_weight"],
    },
    "PCA+SVM": {
        "feat_csv": "PCA_SVM_200_features.csv",
        "prob_csv": "PCA_SVM_200_probabilities.csv",
        "hp_cols":  ["n_components", "kernel", "C", "gamma", "class_weight", "degree"],
    },
}

_EXTRA_LEARNERS = {
    "RF": lambda: RandomForestClassifier(
        n_estimators=500, class_weight="balanced", random_state=_RANDOM_STATE
    ),
    "meta_LR": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=_RANDOM_STATE
    ),
    "meta_PCA+SVM": lambda: Pipeline([
        ("pca", PCA(n_components=0.95, random_state=_RANDOM_STATE)),
        ("svm", SVC(kernel="rbf", C=1.0, class_weight="balanced",
                    probability=False, random_state=_RANDOM_STATE)),
    ]),
    "meta_PCA+RF": lambda: Pipeline([
        ("pca", PCA(n_components=0.95, random_state=_RANDOM_STATE)),
        ("rf",  RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=_RANDOM_STATE
        )),
    ]),
    "meta_PCA+LR": lambda: Pipeline([
        ("pca", PCA(n_components=0.95, random_state=_RANDOM_STATE)),
        ("lr",  LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=_RANDOM_STATE
        )),
    ]),
}


@st.cache_data(show_spinner=False)
def _load_data(path: str):
    df = pd.read_pickle(path)
    return _process_df(df)


@st.cache_data(show_spinner=False)
def _load_data_bytes(data: bytes):
    import io
    df = pd.read_pickle(io.BytesIO(data))
    return _process_df(df)


def _process_df(df):
    df["failureType"] = df["failureType"].apply(
        lambda x: np.array(x).flatten()[0] if np.array(x).size > 0 else None
    )
    df = df.dropna(subset=["failureType"])
    df["failureType"] = df["failureType"].astype(str)
    df = df[df["failureType"] != "none"]
    df = df.dropna(subset=["dieSize"])
    df["dieSize"] = df["dieSize"].astype(int)
    df = df[df["dieSize"] < 2500].reset_index(drop=True)
    params = get_feature_params()
    feat_dicts = df["waferMap"].apply(lambda wm: extract_wafer_features(wm, params))
    feat_df    = pd.DataFrame(feat_dicts.tolist())
    y_str      = df["failureType"].values
    y          = LabelEncoder().fit_transform(y_str)
    # Select top-10 features using SVM importance signal (LinearSVC |coef|)
    top_df, top_names = select_top_features(feat_df, y_str, n_top=10, random_state=42)
    X = top_df.fillna(0.0).replace([np.inf, -np.inf], 0.0).values.astype(float)
    return X, y, top_names, y_str


def _read_best_params(feat_csv_path: str, hp_cols: list) -> dict:
    df  = pd.read_csv(feat_csv_path)
    row = df.iloc[0]
    return {c: row[c] for c in hp_cols if c in row.index}


def _cast_params(model_name: str, raw: dict) -> dict:
    out = {}
    int_keys   = {"max_iter", "n_components", "degree", "n_estimators",
                  "max_depth", "min_samples_split"}
    float_keys = {"C", "gamma", "tol", "reg_param"}
    for k, v in raw.items():
        if k in int_keys:
            out[k] = int(float(v)) if not (isinstance(v, float) and np.isnan(v)) else None
        elif k in float_keys:
            if isinstance(v, float) and np.isnan(v):
                out[k] = None
            else:
                try:
                    out[k] = float(v)
                except (ValueError, TypeError):
                    out[k] = v
        elif k == "shrinkage":
            if isinstance(v, float) and np.isnan(v):
                out[k] = None
            elif isinstance(v, str) and v.lower() == "none":
                out[k] = None
            else:
                out[k] = v
        elif k == "class_weight":
            out[k] = None if (isinstance(v, float) and np.isnan(v)) else str(v)
        elif k == "solver":
            if model_name in ("LR", "PCA+LR"):
                penalty = raw.get("penalty", "l2")
                if str(penalty) == "l1" and str(v) == "lbfgs":
                    out[k] = "saga"
                else:
                    out[k] = str(v)
            else:
                out[k] = str(v)
        elif k == "max_features":
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _build_pipeline(model_name: str, n_comp, params: dict, rng: int):
    use_pca = model_name.startswith("PCA+")
    base_name = model_name[4:] if use_pca else model_name

    if base_name == "LR":
        clf = LogisticRegression(random_state=rng, **params)
    elif base_name == "SVM":
        p = {k: v for k, v in params.items() if k != "gamma"}
        gamma = params.get("gamma", "scale")
        try:
            gamma = float(gamma)
        except (ValueError, TypeError):
            pass
        clf = SVC(gamma=gamma, probability=True, random_state=rng, **p)
    elif base_name == "LDA":
        clf = LinearDiscriminantAnalysis(**params)
    elif base_name == "QDA":
        clf = QuadraticDiscriminantAnalysis(**params)
    else:
        clf = LogisticRegression(random_state=rng)

    steps = [("scaler", StandardScaler())]
    if use_pca and n_comp is not None:
        steps.append(("pca", PCA(n_components=int(n_comp), random_state=rng)))
    steps.append(("clf", clf))
    return Pipeline(steps)


st.title("Stacking Ensemble — Full Training Pipeline")

# ── Dataset input: file uploader (cloud) or local path (local) ──
uploaded_pkl = st.sidebar.file_uploader("Upload LSWMD.pkl", type=["pkl"])
data_path = st.sidebar.text_input("OR local path (.pkl)", value=_DEFAULT_PATH)
test_size  = st.sidebar.slider("Test split fraction", 0.1, 0.4, 0.2, 0.05)
use_smote  = st.sidebar.checkbox("Apply SMOTE to meta-train features", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("**SVM meta-learner grid**")
svm_kernels = st.sidebar.multiselect(
    "Kernels", ["rbf", "linear", "poly", "sigmoid"],
    default=["rbf", "linear", "poly"]
)
svm_cs = st.sidebar.multiselect(
    "C values", [0.01, 0.1, 1.0, 10.0, 100.0],
    default=[0.1, 1.0, 10.0]
)

run_btn = st.sidebar.button("Run Pipeline", type="primary")

if not run_btn:
    st.info("Configure options in the sidebar and click **Run Pipeline**.")
    st.stop()

st.subheader("Step 1 — Load data and split")
with st.spinner("Loading dataset..."):
    if uploaded_pkl is not None:
        X_all, y_all_enc, feat_names, y_all_str = _load_data_bytes(uploaded_pkl.read())
    elif os.path.isfile(data_path):
        X_all, y_all_enc, feat_names, y_all_str = _load_data(data_path)
    else:
        st.error(
            "LSWMD.pkl not found. Please upload the file using the sidebar uploader, "
            "or provide a valid local path.\n\n"
            "Download from: https://www.kaggle.com/datasets/qingyi/wm-811k-wafermap"
        )
        st.stop()

le        = LabelEncoder().fit(np.unique(y_all_str))
y_all     = le.transform(y_all_str)
class_names = list(le.classes_)
n_classes   = len(class_names)

train_idx, test_idx = train_test_split(
    np.arange(len(y_all)),
    test_size=test_size,
    random_state=_RANDOM_STATE,
    stratify=y_all,
)
X_train, X_test = X_all[train_idx], X_all[test_idx]
y_train, y_test = y_all[train_idx], y_all[test_idx]
train_idx_set   = set(train_idx.tolist())

st.success(
    f"Dataset: **{len(y_all)}** samples | "
    f"Train: **{len(train_idx)}** | Test: **{len(test_idx)}** | "
    f"Classes: **{n_classes}**"
)

st.subheader("Step 2 — Train base models and collect test probabilities")
meta_X_test_blocks = []
trained_models     = {}
base_col  = st.columns(min(len(_MODEL_CONFIGS), 4))
col_idx   = 0
model_missing = []

for model_name, cfg in _MODEL_CONFIGS.items():
    feat_csv = os.path.join(_WORK_DIR, cfg["feat_csv"])
    if not os.path.isfile(feat_csv):
        model_missing.append(model_name)
        continue

    raw_params = _read_best_params(feat_csv, cfg["hp_cols"])
    n_comp     = raw_params.pop("n_components", None)
    params     = _cast_params(model_name, raw_params)

    pipe = _build_pipeline(model_name, n_comp, params, _RANDOM_STATE)

    with base_col[col_idx % len(base_col)]:
        with st.spinner(f"Training {model_name}..."):
            pipe.fit(X_train, y_train)
            proba = pipe.predict_proba(X_test)

    trained_models[model_name]  = (pipe, n_comp, params)
    meta_X_test_blocks.append(proba)
    col_idx += 1

if model_missing:
    st.warning(f"Missing feature CSVs for: {model_missing}. Skipped.")

st.subheader("Step 3 — Assemble OOF meta-train features")
oof_blocks     = []
test_blocks_filtered = []
sorted_train_indices = None

for model_name, cfg in _MODEL_CONFIGS.items():
    if model_name not in trained_models:
        continue
    prob_csv = os.path.join(_WORK_DIR, cfg["prob_csv"])
    if not os.path.isfile(prob_csv):
        st.warning(f"OOF CSV not found: {cfg['prob_csv']}. Skipping {model_name}.")
        continue

    df_oof    = pd.read_csv(prob_csv)
    df_train  = df_oof[df_oof["sample_idx"].isin(train_idx_set)].copy()
    df_train  = df_train.sort_values("sample_idx").reset_index(drop=True)
    prob_cols = [c for c in df_train.columns if c.startswith("prob_")]

    if sorted_train_indices is None:
        sorted_train_indices = df_train["sample_idx"].values.astype(int)

    oof_blocks.append(df_train[prob_cols].values.astype(float))
    # Keep only test blocks for models that also have OOF CSVs
    model_list = list(_MODEL_CONFIGS.keys())
    model_idx  = [i for i, m in enumerate(model_list) if m in trained_models]
    oof_model_order = [m for m in model_list if m in trained_models and
                       os.path.isfile(os.path.join(_WORK_DIR, _MODEL_CONFIGS[m]["prob_csv"]))]

# Rebuild meta_X_test using only models that contributed to oof_blocks
test_blocks_filtered = []
for model_name in _MODEL_CONFIGS:
    if model_name not in trained_models:
        continue
    prob_csv = os.path.join(_WORK_DIR, _MODEL_CONFIGS[model_name]["prob_csv"])
    if not os.path.isfile(prob_csv):
        continue
    pipe, _, _ = trained_models[model_name]
    test_blocks_filtered.append(pipe.predict_proba(X_test))

if not oof_blocks:
    st.error("No OOF CSVs found. Cannot build meta_X_train.")
    st.stop()

meta_X_train    = np.hstack(oof_blocks)
meta_X_test     = np.hstack(test_blocks_filtered)
y_meta_train    = y_all[sorted_train_indices]

st.success(
    f"meta_X_train shape: **{meta_X_train.shape}**  |  "
    f"meta_X_test shape: **{meta_X_test.shape}**"
)

st.subheader("Step 4 — Scale meta-features")
meta_scaler      = StandardScaler()
meta_X_train_sc  = meta_scaler.fit_transform(meta_X_train)
meta_X_test_sc   = meta_scaler.transform(meta_X_test)

if use_smote:
    with st.spinner("Applying SMOTE..."):
        smote = SMOTE(random_state=_RANDOM_STATE)
        meta_X_train_sc, y_meta_train = smote.fit_resample(meta_X_train_sc, y_meta_train)
    st.info(f"After SMOTE: meta_X_train shape = **{meta_X_train_sc.shape}**")

st.subheader("Step 5 — Fit and evaluate meta-learners")


def _eval_meta(name, clf, X_tr, y_tr, X_te, y_te):
    clf.fit(X_tr, y_tr)
    pred     = clf.predict(X_te)
    bal_acc  = balanced_accuracy_score(y_te, pred)
    acc      = accuracy_score(y_te, pred)
    prec_mac = precision_score(y_te, pred, average="macro", zero_division=0)
    rec_mac  = recall_score(y_te, pred, average="macro", zero_division=0)
    f1_mac   = f1_score(y_te, pred, average="macro", zero_division=0)
    per_cls  = {}
    for ci, cn in enumerate(class_names):
        per_cls[cn] = {
            "precision": precision_score(y_te, pred,
                                         labels=[ci], average="macro", zero_division=0),
            "recall":    recall_score(y_te, pred,
                                      labels=[ci], average="macro", zero_division=0),
            "f1":        f1_score(y_te, pred,
                                  labels=[ci], average="macro", zero_division=0),
            "support":   int((y_te == ci).sum()),
        }
    return {
        "name": name, "pred": pred,
        "bal_acc": bal_acc, "acc": acc,
        "prec_mac": prec_mac, "rec_mac": rec_mac, "f1_mac": f1_mac,
        "per_class": per_cls,
    }


all_results = []

if not svm_kernels or not svm_cs:
    st.warning("Select at least one SVM kernel and C value from the sidebar.")
else:
    svm_bar = st.progress(0.0)
    total   = len(svm_kernels) * len(svm_cs)
    done    = 0
    for kernel in svm_kernels:
        for C_val in svm_cs:
            label = f"SVM  kernel={kernel}  C={C_val}"
            clf   = SVC(kernel=kernel, C=float(C_val),
                        class_weight="balanced", probability=False,
                        random_state=_RANDOM_STATE)
            r = _eval_meta(label, clf, meta_X_train_sc, y_meta_train,
                           meta_X_test_sc, y_test)
            all_results.append(r)
            done += 1
            svm_bar.progress(done / total)
    svm_bar.empty()

extra_bar = st.progress(0.0)
extra_names = list(_EXTRA_LEARNERS.keys())
for i, name in enumerate(extra_names):
    clf = _EXTRA_LEARNERS[name]()
    r   = _eval_meta(name, clf, meta_X_train_sc, y_meta_train,
                     meta_X_test_sc, y_test)
    all_results.append(r)
    extra_bar.progress((i + 1) / len(extra_names))
extra_bar.empty()

all_results.sort(key=lambda r: r["bal_acc"], reverse=True)
best_name = all_results[0]["name"]

st.subheader("Results Summary")
summary_rows = []
for r in all_results:
    label = f"{'★ ' if r['name'] == best_name else ''}{r['name']}"
    summary_rows.append({
        "Meta-learner":   label,
        "Balanced Acc":   round(r["bal_acc"], 4),
        "Accuracy":       round(r["acc"], 4),
        "Precision (mac)": round(r["prec_mac"], 4),
        "Recall (mac)":   round(r["rec_mac"], 4),
        "F1 (mac)":       round(r["f1_mac"], 4),
    })
summary_df = pd.DataFrame(summary_rows)
st.dataframe(summary_df.set_index("Meta-learner"), use_container_width=True)

csv_summary = summary_df.to_csv(index=False).encode()
st.download_button(
    "Download summary CSV", data=csv_summary,
    file_name="stacking_results_summary.csv", mime="text/csv",
)

st.subheader("Per-Learner Details")
for r in all_results:
    is_best = r["name"] == best_name
    label   = f"{'★ ' if is_best else ''}{r['name']}  |  Balanced Acc = {r['bal_acc']:.4f}"
    with st.expander(label, expanded=is_best):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Balanced Acc",   f"{r['bal_acc']:.4f}")
        c2.metric("Accuracy",       f"{r['acc']:.4f}")
        c3.metric("F1 (macro)",     f"{r['f1_mac']:.4f}")
        c4.metric("Precision (mac)", f"{r['prec_mac']:.4f}")

        pc_rows = []
        for cn, m in r["per_class"].items():
            pc_rows.append({
                "Class":     cn,
                "Precision": round(m["precision"], 4),
                "Recall":    round(m["recall"], 4),
                "F1":        round(m["f1"], 4),
                "Support":   m["support"],
            })
        pc_df = pd.DataFrame(pc_rows).set_index("Class")
        st.dataframe(pc_df, use_container_width=True)

        dl_key = f"dl_eat_{r['name'].replace(' ', '_')}"
        csv_pc = pc_df.reset_index().to_csv(index=False).encode()
        st.download_button(
            f"Download {r['name']} per-class CSV",
            data=csv_pc,
            file_name=f"stacking_{r['name'].replace(' ', '_')}_per_class.csv",
            mime="text/csv",
            key=dl_key,
        )

        cm  = confusion_matrix(y_test, r["pred"])
        fig = px.imshow(
            cm, text_auto=True, aspect="auto",
            x=class_names, y=class_names,
            labels={"x": "Predicted", "y": "True", "color": "Count"},
            title=f"{r['name']} — Confusion Matrix (test set)",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Best hyperparameters used per base model"):
    hp_rows = []
    for model_name, (pipe, n_comp, params) in trained_models.items():
        row = {"Model": model_name}
        if n_comp is not None:
            row["n_components"] = n_comp
        row.update(params)
        hp_rows.append(row)
    if hp_rows:
        st.dataframe(pd.DataFrame(hp_rows).set_index("Model"), use_container_width=True)
