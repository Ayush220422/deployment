import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import preprocessor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def plotter(x, y, y_pred=None, poly=None):
    fig, ax = plt.subplots(figsize=(1, 1))
    ax.set_title('2D- Data plotted', fontsize=5)
    ax.set_xlabel('inputs', fontsize=2)
    ax.set_ylabel('outputs', fontsize=5)

    if poly == None:
        fig = ax.plot(x, y, color='blue', marker='.')
    elif poly == 0:
        fig = ax.plot(x, y, color='blue', marker='.')
        fig = ax.plot(x, y_pred, color='red', marker='red')
    elif poly = 1:
        fig = ax.plot(x, y, color='blue', marker='.')
        fig = ax.plot(x, y_pred, color='red', marker='r.')

    return fig


def model_train(model_type):
    x = df[:, 0:1]
    y = df[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=2)
    model = LinearRegression()

    if model_type == "Linear Regression":
        model.fit(x_train, y_train)
        y_pred = model.predict(x)
        y_pred1 = model.predict(x_test)
    #   r2_Score = r2_score(y_test, y_pred1)

        return y_pred, r2_score(y_test, y_pred1)
    elif model_type == "Polynomial Regression":
        poly_value = st.slider("Degree", 0, 20, 2)
        poly = PolynomialFeatures(degree=poly_value)

        x_train_trans = poly.fit_transform(x_train)
        x_test_trans = poly.transform(x_test)
        model.fit(x_train_trans, y_train)
        y_pred = model.predict(x)
        y_pred1 = model.predict(x_test_trans)
    #   r2_Score = r2_score(y_test, y_pred1)

        return y_pred, r2_score(y_test, y_pred1)

    def runner(uploaded_file):
        run = st.button("Run Algorithm")

        if run:
            if (uploaded_file) == None:
                st.error("Please choose a File first!!!")
