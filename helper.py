import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


def plotter(x, y, y_pred=None, poly=None):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_title('2D- Data plotted', fontsize=5)
    ax.set_xlabel('inputs', fontsize=2)
    ax.set_ylabel('outputs', fontsize=5)
    x = x.values.ravel()
    y = y.values.ravel()

    if poly == None:
        ax.plot(x, y, 'b.')
    elif poly == 0:
        ax.scatter(x, y, color='blue', marker='.')
        ax.plot(x, y_pred, color='red')
    elif poly == 1:
        ax.plot(x, y, 'b.')
        ax.plot(x, y_pred,'r.')

    return fig


def model_train(df, model_type, poly_value=None):
    x = df.iloc[:, 0:1]
    y = df.iloc[:, -1:]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=2)

    if model_type == "Linear Regression":
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x)
        y_pred1 = model.predict(x_test)
        r2_Score1 = r2_score(y_test, y_pred1)

        return y_pred, r2_Score1
    elif model_type == "Polynomial Regression":
        model = LinearRegression()
        poly = PolynomialFeatures(degree=poly_value)

        x_train_trans = poly.fit_transform(x_train)
        x_test_trans = poly.transform(x_test)
        x_full_trans = poly.transform(x)
        model.fit(x_train_trans, y_train)
        y_pred = model.predict(x_full_trans)
        y_pred1 = model.predict(x_test_trans)
        r2_Score1 = r2_score(y_test, y_pred1)

        return y_pred, r2_Score1
