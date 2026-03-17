import plotly.express as px
import streamlit as st
import numpy as np

def plot_maps(df):

    categories = df["failureType"].unique()

    cols = st.columns(3)

    for i, cat in enumerate(categories):

        wafer = np.array(df[df["failureType"] == cat].iloc[0]["waferMap"])

        fig = px.imshow(
            wafer,
            color_continuous_scale=[
                (0.0, "white"),       # empty
                (0.5, "lightgreen"),  # good
                (1.0, "maroon")       # bad
            ],
            zmin=0,
            zmax=2
        )

        fig.update_layout(
            width=190,
            height=190,
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_showscale=False
        )

        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        with cols[i % 3]:
            st.plotly_chart(fig, use_container_width=False)
            st.markdown(
                f"<div style='text-align:center'>{cat}</div>",
                unsafe_allow_html=True
            )
 