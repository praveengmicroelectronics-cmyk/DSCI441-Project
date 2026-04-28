import os
import glob
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

_WORK_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Model Performance Analysis", layout="wide")
st.title("Feature CSV — Iteration Performance Analysis")

# Auto-load all *_features.csv from the repo directory
csv_paths = sorted(glob.glob(os.path.join(_WORK_DIR, "*_features.csv")))

# Optional: allow uploading additional CSVs
uploaded_csvs = st.sidebar.file_uploader(
    "Upload additional *_features.csv files (optional)",
    type=["csv"], accept_multiple_files=True
)

if not csv_paths and not uploaded_csvs:
    st.error("No *_features.csv files found in the repo directory and none uploaded.")
    st.stop()


def _label(path):
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem[: -len("_features")] if stem.endswith("_features") else stem


def _load(path):
    df = pd.read_csv(path)
    metric_start = df.columns.get_loc("balanced_acc") if "balanced_acc" in df.columns else 0
    hp_cols = list(df.columns[:metric_start])
    df["_iter"]  = df.index
    df["_label"] = df[hp_cols].fillna("—").astype(str).agg(" | ".join, axis=1) if hp_cols else df.index.astype(str)
    return df, hp_cols


dfs_all, hp_cols_all = {}, {}
for path in csv_paths:
    name = _label(path)
    try:
        df, hc = _load(path)
        dfs_all[name]     = df
        hp_cols_all[name] = hc
    except Exception as e:
        st.warning(f"Could not load {os.path.basename(path)}: {e}")

for uf in (uploaded_csvs or []):
    name = _label(uf.name)
    try:
        import io
        df, hc = _load(io.StringIO(uf.read().decode("utf-8")))
        dfs_all[name]     = df
        hp_cols_all[name] = hc
    except Exception as e:
        st.warning(f"Could not load uploaded {uf.name}: {e}")

if not dfs_all:
    st.stop()

all_names = sorted(dfs_all.keys())
selected  = st.sidebar.multiselect("Models to analyse", all_names, default=all_names)

if not selected:
    st.warning("Select at least one model from the sidebar.")
    st.stop()


st.subheader("Cross-Model Comparison — Best Balanced Accuracy")
cmp_rows = []
for name in all_names:
    df = dfs_all[name]
    if "balanced_acc" not in df.columns:
        continue
    gap_col = "E_train-E_test"
    cmp_rows.append({
        "Model":                    name,
        "Iterations":               len(df),
        "Best Balanced Acc":        round(float(df["balanced_acc"].max()), 4),
        "Mean Balanced Acc":        round(float(df["balanced_acc"].mean()), 4),
        "E_train-E_test @ best":    round(float(df.loc[df["balanced_acc"].idxmax(), gap_col]), 4)
                                    if gap_col in df.columns else None,
    })
cmp_df = pd.DataFrame(cmp_rows).sort_values("Best Balanced Acc", ascending=False).set_index("Model")
st.dataframe(cmp_df, use_container_width=True)

fig_cmp = px.bar(
    cmp_df.reset_index().sort_values("Best Balanced Acc"),
    x="Best Balanced Acc", y="Model", orientation="h",
    title="Best Balanced Accuracy per Model",
    color="Best Balanced Acc", color_continuous_scale="Blues",
    text="Best Balanced Acc",
)
fig_cmp.update_traces(textposition="outside")
fig_cmp.update_layout(coloraxis_showscale=False, height=max(300, 60 * len(cmp_rows)))
st.plotly_chart(fig_cmp, use_container_width=True)


st.subheader("Bias–Variance Analysis — E_train / E_test / |Gap| at Best Config")
bv_rows = []
for name in all_names:
    df = dfs_all[name]
    if "balanced_acc" not in df.columns:
        continue
    best_row = df.loc[df["balanced_acc"].idxmax()]
    e_train  = float(best_row["E_train"])        if "E_train"        in df.columns else None
    e_test   = float(best_row["E_test"])         if "E_test"         in df.columns else None
    gap_raw  = float(best_row["E_train-E_test"]) if "E_train-E_test" in df.columns else None
    bv_rows.append({
        "Model":          name,
        "E_train":        round(e_train, 4)          if e_train  is not None else None,
        "E_test":         round(e_test,  4)          if e_test   is not None else None,
        "abs_gap":        round(abs(gap_raw), 4)     if gap_raw  is not None else None,
    })

if bv_rows:
    # Sort ascending by |gap|: low-gap (bias-dominated) on left, high-gap (variance-dominated) on right
    bv_df = pd.DataFrame(bv_rows).sort_values("abs_gap", ascending=True)
    fig_bv = go.Figure()
    fig_bv.add_trace(go.Scatter(
        x=bv_df["Model"], y=bv_df["E_train"], mode="lines+markers",
        name="E_train", line=dict(color="royalblue", width=2),
        marker=dict(size=8),
    ))
    fig_bv.add_trace(go.Scatter(
        x=bv_df["Model"], y=bv_df["E_test"], mode="lines+markers",
        name="E_test", line=dict(color="orange", width=2),
        marker=dict(size=8),
    ))
    fig_bv.add_trace(go.Scatter(
        x=bv_df["Model"], y=bv_df["abs_gap"], mode="lines+markers",
        name="|E_train−E_test| (gap)", line=dict(color="tomato", width=2, dash="dot"),
        marker=dict(size=8, symbol="diamond"),
    ))
    fig_bv.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.4)
    fig_bv.update_layout(
        title="Bias–Variance: E_train / E_test / |Gap| at Best Config  ←bias  |  variance→",
        xaxis_title="Model  (sorted by |gap| ascending — low variance left, high variance right)",
        yaxis_title="Error",
        legend=dict(orientation="h", y=-0.25),
        height=460,
        xaxis_tickangle=-30,
    )
    st.plotly_chart(fig_bv, use_container_width=True)


for name in selected:
    df      = dfs_all[name]
    hp_cols = hp_cols_all[name]
    st.markdown("---")
    st.subheader(name)

    best_idx    = int(df["balanced_acc"].idxmax()) if "balanced_acc" in df.columns else 0
    best_params = {c: df.loc[best_idx, c] for c in hp_cols} if hp_cols else {}
    st.caption(
        f"{len(df)} iterations  |  "
        f"Best Balanced Acc: **{df['balanced_acc'].max():.4f}** at iteration {best_idx}  |  "
        f"Best params: {best_params}"
    )

    col1, col2 = st.columns(2)

    with col1:
        if "balanced_acc" in df.columns:
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=df["_iter"], y=df["balanced_acc"], mode="lines+markers",
                name="balanced_acc", hovertext=df["_label"],
                line=dict(color="royalblue"),
            ))
            fig_acc.add_vline(x=best_idx, line_dash="dash", line_color="red",
                              annotation_text="best", annotation_position="top right")
            fig_acc.update_layout(
                title="Balanced Accuracy vs Iteration",
                xaxis_title="Iteration", yaxis_title="Balanced Accuracy",
                height=380,
            )
            st.plotly_chart(fig_acc, use_container_width=True)

    with col2:
        gap_col = "E_train-E_test"
        if gap_col in df.columns:
            fig_gap = go.Figure()
            if "E_train" in df.columns:
                fig_gap.add_trace(go.Scatter(
                    x=df["_iter"], y=df["E_train"], mode="lines+markers",
                    name="E_train", hovertext=df["_label"],
                    line=dict(color="royalblue"),
                ))
            if "E_test" in df.columns:
                fig_gap.add_trace(go.Scatter(
                    x=df["_iter"], y=df["E_test"], mode="lines+markers",
                    name="E_test", hovertext=df["_label"],
                    line=dict(color="orange"),
                ))
            fig_gap.add_trace(go.Scatter(
                x=df["_iter"], y=df[gap_col], mode="lines+markers",
                name="E_train−E_test", hovertext=df["_label"],
                line=dict(color="tomato", dash="dot"),
            ))
            fig_gap.add_hline(y=0, line_dash="dash", line_color="black", line_width=1, opacity=0.4)
            fig_gap.add_vline(x=best_idx, line_dash="dash", line_color="red",
                              annotation_text="best", annotation_position="top right")
            fig_gap.update_layout(
                title="E_train / E_test / Generalisation Gap vs Iteration",
                xaxis_title="Iteration", yaxis_title="Error",
                legend=dict(orientation="h", y=-0.2),
                height=400,
            )
            st.plotly_chart(fig_gap, use_container_width=True)

    gap_col = "E_train-E_test"
    if gap_col in df.columns and "balanced_acc" in df.columns:
        fig_scatter = px.scatter(
            df, x=gap_col, y="balanced_acc",
            hover_data=hp_cols if hp_cols else None,
            title=f"{name} — E_train−E_test vs Balanced Accuracy",
            labels={gap_col: "E_train − E_test", "balanced_acc": "Balanced Accuracy"},
            color="balanced_acc", color_continuous_scale="Viridis",
        )
        fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1)
        best_row = df.loc[best_idx]
        fig_scatter.add_trace(go.Scatter(
            x=[best_row[gap_col]], y=[best_row["balanced_acc"]],
            mode="markers", marker=dict(size=14, color="red", symbol="star"),
            name="Best", showlegend=True,
        ))
        fig_scatter.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(fig_scatter, use_container_width=True)
