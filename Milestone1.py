import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    confusion_matrix,
)
from sklearn.exceptions import ConvergenceWarning

from plot_maps import plot_maps
from Grid_Generation import get_grid
from Feature_parameters import get_feature_params
from Feature_Extraction import extract_wafer_features


path = os.path.expanduser("~/Documents/DSCI441/DSCI441-Project/LSWMD.pkl")
df = pd.read_pickle(path)
df["failureType"] = df["failureType"].apply(lambda x: np.array(x).flatten()[0] if np.array(x).size > 0 else None)
df = df.dropna(subset=["failureType"])
df["failureType"] = df["failureType"].astype("category")
df = df[df["failureType"] != "none"]
#print(df.shape)
#print(df["failureType"].value_counts())
#print(df["dieSize"].value_counts())
df = df.dropna(subset=["dieSize"])
df["dieSize"] = df["dieSize"].astype(int)
st.write("Data loaded")
st.write(df.shape)
plot_maps(df)
st.markdown(
    "**Legend:** "
    "<span style='color:lightgreen'>■</span> Good die &nbsp;&nbsp; "
    "<span style='color:maroon'>■</span> Bad die &nbsp;&nbsp; "
    "<span style='color:white; border:1px solid black'>■</span> Empty",
    unsafe_allow_html=True
)
# Sidebar slider
st.sidebar.header("Inputs")
ds_min = int(df["dieSize"].min())
ds_max = int(df["dieSize"].max())
ds_default = int(df["dieSize"].median())

die_min = st.sidebar.slider(
    "Minimum die size (min dimension)",
    min_value=ds_min,
    max_value=ds_max,
    value=ds_default,
    step=1,
)


# Filter by die size
df_filtered = df[df["dieSize"] >= die_min].copy()
st.sidebar.markdown("### Filtered dataset")
st.sidebar.write(f"Rows: {len(df_filtered):,}")
st.sidebar.dataframe(df_filtered["failureType"].value_counts().rename("count").to_frame())

st.write("### Dataset after filtering")
st.write(df_filtered.shape)


params=get_feature_params()
df = df.dropna(subset=["failureType"])
features = df_filtered["waferMap"].apply(
    lambda wm: extract_wafer_features(wm, params)
)
features_df = pd.DataFrame(features.tolist())
st.write(features_df.shape)


##########################################################

###########Mode Training ##############################

##########################################################

X = features_df
y = df_filtered["failureType"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)

model = RandomForestClassifier()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

st.write("ACCCURACY for Baseline")
st.write(f"{accuracy_score(y_test, y_pred) * 100:.2f}%")
st.write("Balanced Accuracy for Baseline")
st.write(f"{balanced_accuracy_score(y_test, y_pred) * 100:.2f}%")

#print(f1_score(y_test, y_pred))
st.write("RECALL SCORE for Baseline")
recalls=recall_score(y_test, y_pred, average=None)
recall_df = pd.DataFrame({
    "Defect Type": model.classes_,
    "Recall (%)": [f"{r*100:.2f}%" for r in recalls]
})

st.table(recall_df)

labels = model.classes_

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)
confusion_matrix(y_test, y_pred).tolist()





























