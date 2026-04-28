import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from Ensemble_function import validate_csv_files, load_meta_features

_DEFAULT_DIR = os.path.dirname(os.path.abspath(__file__))
_META_LEARNERS = {
    "LR": lambda: LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=42
    ),
    "RF": lambda: RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=1
    ),
    "SVM": lambda: LinearSVC(
        C=1.0, class_weight="balanced", max_iter=2000, random_state=42
    ),
    "LDA": lambda: LinearDiscriminantAnalysis(),
    "QDA": lambda: QuadraticDiscriminantAnalysis(reg_param=0.01),
}


@st.cache_data(show_spinner=False)
def _load(search_dir):
    return load_meta_features(search_dir=search_dir, run_validation=False)


st.set_page_config(page_title="Ensemble Validation", layout="wide")
st.title("Stacking Ensemble — OOF Cross-Validation Evaluation")

search_dir = st.sidebar.text_input("Probability CSV directory", value=_DEFAULT_DIR)
n_folds    = st.sidebar.slider("CV folds (k)", min_value=3, max_value=10, value=5)
selected   = st.sidebar.multiselect(
    "Meta-learners to evaluate",
    options=list(_META_LEARNERS.keys()),
    default=list(_META_LEARNERS.keys()),
)

st.sidebar.markdown("---")
st.sidebar.markdown("Run `streamlit run Ensemble_Validation.py`")

st.subheader("Step 0 — Probability CSV Validation")
with st.spinner("Validating probability CSVs..."):
    report = validate_csv_files(search_dir=search_dir)

if report["passed"]:
    st.success(report["summary"])
else:
    st.error(report["summary"])

with st.expander("Per-file validation details", expanded=not report["passed"]):
    rows = []
    for model, chk in report["per_file"].items():
        rows.append({
            "Model":          model,
            "Rows":           chk.get("n_rows", "N/A"),
            "Prob cols":      chk.get("n_prob_cols", "N/A"),
            "Duplicates":     chk.get("duplicate_sample_idx", "N/A"),
            "NaN":            chk.get("n_nan_in_probs", "N/A"),
            "Inf":            chk.get("n_inf_in_probs", "N/A"),
            "Bad sum rows":   chk.get("n_rows_bad_prob_sum", "N/A"),
            "Passed":         chk.get("file_passed", False),
        })
    st.dataframe(pd.DataFrame(rows).set_index("Model"), width="stretch")

cross = report["cross_file"]
if "skipped" not in cross:
    with st.expander("Cross-file validation details", expanded=not report["passed"]):
        cross_rows = [
            {"Check": "Row counts consistent",   "Result": cross.get("row_counts_ok", "N/A")},
            {"Check": "Class order consistent",  "Result": cross.get("class_order_ok", "N/A")},
            {"Check": "sample_idx set OK",       "Result": cross.get("sample_idx_set_ok", "N/A")},
            {"Check": "sample_idx order OK",     "Result": cross.get("sample_idx_order_ok", "N/A")},
            {"Check": "true_label consistent",   "Result": cross.get("true_label_ok", "N/A")},
        ]
        st.dataframe(pd.DataFrame(cross_rows).set_index("Check"), width="stretch")

if not report["passed"]:
    st.stop()

@st.cache_data(show_spinner=False)
def _run_cv(meta_X, y_arr, class_names_tuple, selected_tuple, n_folds):
    le      = LabelEncoder().fit(list(class_names_tuple))
    y       = y_arr
    skf     = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []
    for name in selected_tuple:
        clf  = _META_LEARNERS[name]()
        pred = cross_val_predict(clf, meta_X, y, cv=skf, n_jobs=1)

        clf_full = _META_LEARNERS[name]()
        clf_full.fit(meta_X, y)
        train_pred    = clf_full.predict(meta_X)
        e_train = round(1.0 - balanced_accuracy_score(y, train_pred), 4)
        e_test  = round(1.0 - balanced_accuracy_score(y, pred), 4)

        per_class_macro = {}
        for cls_name in le.classes_:
            lbl = le.transform([cls_name])[0]
            per_class_macro[cls_name] = {
                "precision": precision_score(y, pred, labels=[lbl], average="macro", zero_division=0),
                "recall":    recall_score(y, pred, labels=[lbl], average="macro", zero_division=0),
                "f1":        f1_score(y, pred, labels=[lbl], average="macro", zero_division=0),
                "support":   int((y == lbl).sum()),
            }

        results.append({
            "name":      name,
            "bal_acc":   balanced_accuracy_score(y, pred),
            "acc":       accuracy_score(y, pred),
            "prec_mac":  precision_score(y, pred, average="macro", zero_division=0),
            "rec_mac":   recall_score(y, pred, average="macro", zero_division=0),
            "f1_mac":    f1_score(y, pred, average="macro", zero_division=0),
            "e_train":   e_train,
            "e_test":    e_test,
            "abs_gap":   round(abs(e_train - e_test), 4),
            "pred":      pred,
            "per_class": per_class_macro,
        })
    return results


st.subheader("Step 1 — Load Meta-Feature Matrix")
with st.spinner("Loading probability CSVs..."):
    meta_X, y_str, model_names, class_names = _load(search_dir)

le = LabelEncoder().fit(class_names)
y  = le.transform(y_str)

st.success(
    f"meta_X shape: **{meta_X.shape}**  "
    f"({len(model_names)} base models x {len(class_names)} classes)  |  "
    f"N samples: **{len(y)}**"
)

if not selected:
    st.warning("Select at least one meta-learner from the sidebar.")
    st.stop()

st.subheader("Step 2 — Cross-Validation")
with st.spinner("Running cross-validation (cached after first run)..."):
    all_results = _run_cv(meta_X, y, tuple(class_names), tuple(selected), n_folds)


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
st.dataframe(summary_df.set_index("Meta-learner"), width="stretch")

csv_summary = summary_df.to_csv(index=False).encode()
st.download_button(
    "Download summary CSV", data=csv_summary,
    file_name="ensemble_validation_summary.csv", mime="text/csv",
)

col_g1, col_g2 = st.columns(2)

with col_g1:
    fig_bacc = go.Figure()
    sorted_res = sorted(all_results, key=lambda r: r["bal_acc"])
    fig_bacc.add_trace(go.Scatter(
        x=[r["name"] for r in sorted_res],
        y=[r["bal_acc"] for r in sorted_res],
        mode="lines+markers", name="Balanced Acc",
        line=dict(color="royalblue", width=2), marker=dict(size=9),
    ))
    fig_bacc.update_layout(
        title="Balanced Accuracy per Meta-Learner",
        xaxis_title="Meta-Learner (sorted ascending)",
        yaxis_title="Balanced Accuracy",
        xaxis_tickangle=-30, height=420,
    )
    st.plotly_chart(fig_bacc, width="stretch")

with col_g2:
    sorted_bv = sorted(all_results, key=lambda r: r["abs_gap"])
    fig_bv = go.Figure()
    fig_bv.add_trace(go.Scatter(
        x=[r["name"] for r in sorted_bv],
        y=[r["e_train"] for r in sorted_bv],
        mode="lines+markers", name="E_train",
        line=dict(color="royalblue", width=2), marker=dict(size=8),
    ))
    fig_bv.add_trace(go.Scatter(
        x=[r["name"] for r in sorted_bv],
        y=[r["e_test"] for r in sorted_bv],
        mode="lines+markers", name="E_test",
        line=dict(color="orange", width=2), marker=dict(size=8),
    ))
    fig_bv.add_trace(go.Scatter(
        x=[r["name"] for r in sorted_bv],
        y=[r["abs_gap"] for r in sorted_bv],
        mode="lines+markers", name="|E_train−E_test| (gap)",
        line=dict(color="tomato", width=2, dash="dot"), marker=dict(size=8, symbol="diamond"),
    ))
    fig_bv.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.4)
    fig_bv.update_layout(
        title="Bias–Variance: E_train / E_test / |Gap|  ←bias | variance→",
        xaxis_title="Meta-Learner (sorted by |gap| ascending)",
        yaxis_title="Error (1 − Balanced Acc)",
        legend=dict(orientation="h", y=-0.3),
        xaxis_tickangle=-30, height=420,
    )
    st.plotly_chart(fig_bv, width="stretch")

st.subheader("Per-Learner Details")
for r in all_results:
    is_best = r["name"] == best_name
    label   = f"{'★ ' if is_best else ''}{r['name']}  |  Balanced Acc = {r['bal_acc']:.4f}"
    with st.expander(label, expanded=is_best):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Balanced Acc", f"{r['bal_acc']:.4f}")
        c2.metric("Accuracy",     f"{r['acc']:.4f}")
        c3.metric("F1 (macro)",   f"{r['f1_mac']:.4f}")
        c4.metric("Precision (macro)", f"{r['prec_mac']:.4f}")

        pc_rows = []
        for cls_name, metrics in r["per_class"].items():
            pc_rows.append({
                "Class":     cls_name,
                "Precision": round(metrics["precision"], 4),
                "Recall":    round(metrics["recall"], 4),
                "F1":        round(metrics["f1"], 4),
                "Support":   metrics["support"],
            })
        pc_df = pd.DataFrame(pc_rows).set_index("Class")
        st.dataframe(pc_df, width="stretch")

        csv_pc = pc_df.reset_index().to_csv(index=False).encode()
        st.download_button(
            f"Download {r['name']} per-class CSV",
            data=csv_pc,
            file_name=f"ensemble_val_{r['name']}_per_class.csv",
            mime="text/csv",
            key=f"dl_{r['name']}",
        )

        cm = confusion_matrix(y, r["pred"])
        fig = px.imshow(
            cm, text_auto=True, aspect="auto",
            x=list(le.classes_), y=list(le.classes_),
            labels={"x": "Predicted", "y": "True", "color": "Count"},
            title=f"{r['name']} — Confusion Matrix ({n_folds}-fold OOF)",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, width="stretch")
