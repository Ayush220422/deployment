import streamlit as st
import numpy as np
import pandas as pd
from helper import plotter
from helper import model_train

st.sidebar.title("Linear and Polynomial Regression")

uploaded_file = st.sidebar.file_uploader('choose a file')

run = st.sidebar.button("Run Algorithm")
model_type = st.selectbox(
    "Choose Model",
    ["Linear Regression", "Polynomial Regression"]
)
if model_type == "Polynomial Regression":
        poly_value = st.slider("Degree", 0, 20, 2)

if run:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        x = df.iloc[:, 0:1]
        y = df.iloc[:, -1]
        img = plotter(x, y)
        if model_type == "Linear Regression":
            y_pred, r2_score = model_train(df,model_type)
            img1 = plotter(x, y, y_pred, 0)
        if model_type == "Polynomial Regression":
            y_pred, r2_score = model_train(df,model_type,poly_value)
            img1 = plotter(x, y, y_pred, 1)
    else:
        st.error("Please Enter a File First!!!")
