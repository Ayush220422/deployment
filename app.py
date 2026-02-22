import streamlit as st
import numpy as np
import pandas as pd
from helper import plotter

st.sidebar.title("Linear Regression")

uploaded_file = st.sidebar.file_uploader('choose a file')

run = st.sidebar.button("Run Algorithm")
if run:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        x = df[:, 0:1]
        y = df[:, -1]
        img = helper.plotter(x, y)
        model_type = st.selectbox(
            "Choose Model",
            ["Linear Regression", "Polynomial Regression"]
        )
        y_pred, r2_score = helper.model_train(model_type)
        if model_type == "Linear Regression":
            img1 = helper.plotter(x, y, y_pred, 0)
        if model_type == "Polynomial Regression":
            img1 = helper.plotter(x, y, y_pred, 1)
    else:
        st.error("Please Enter a File First!!!")

