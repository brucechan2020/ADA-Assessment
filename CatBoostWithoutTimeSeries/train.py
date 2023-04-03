import os
import pandas as pd
import numpy as np
import math
import matplotlib.dates as md
from matplotlib.dates import AutoDateLocator
from matplotlib.dates import MONTHLY
from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split

from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import seaborn as sns
import matplotlib.pyplot as plt
from colorama import Fore
import optuna
from optuna.samplers import TPESampler

testSize = 0.1

def readData(fileName):
    df = pd.read_csv(os.path.dirname(
        __file__) + "/" + fileName, encoding='unicode_escape')
    return df


def objective(trial):
    X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=testSize)
    param = {
        "iterations":trial.suggest_int("iterations", 100, 10000),
        "loss_function": trial.suggest_categorical("loss_function", ["RMSE", "MAE"]),
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 10),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 1e-2, 1e0),
        "depth": trial.suggest_int("depth", 3, 10),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 20),
    }

    reg = CatBoostRegressor(**param, cat_features=categorical_features_indices)
    reg.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0, early_stopping_rounds=100)
    y_pred = reg.predict(X_test)
    score = r2_score(y_test, y_pred)
    
    return score


if __name__ == "__main__":
    df = readData("train_data.csv")
    
    df['Date'] = pd.to_datetime(df.Date)
    time = df['Date']
    df.set_index('Date', inplace=True, drop=True)
    
    dataX = df.drop(['BTC_Close'], axis=1)
    dataY = df['BTC_Close']
    categorical_features_indices = np.where(dataX.dtypes != np.float64)[0]
    study = optuna.create_study(sampler=TPESampler(), direction="maximize")
    study.optimize(objective, n_trials=30, timeout=60)
    fig = optuna.visualization.plot_param_importances(study)
    # fig.show()

    

    scaler = MinMaxScaler(feature_range=(0, 1))
    dataY = scaler.fit_transform(np.array(dataY).reshape(-1, 1))
    trainX, testX, trainY, testY = train_test_split(
        dataX, dataY, test_size=testSize, shuffle=False)

    model = CatBoostRegressor(**study.best_params)
    model.fit(trainX, trainY)
    pred = model.predict(testX)
    print("Mean Absolute Error - MAE : " +
          str(mean_absolute_error(testY, pred)))
    print("Root Mean squared Error - RMSE : " +
          str(math.sqrt(mean_squared_error(testY, pred))))

    predict = model.predict(dataX)
    predict = scaler.inverse_transform(predict.reshape(-1, 1))
    fig = plt.figure(figsize=(15, 8))
    
    ax = fig.add_subplot()
    plt.title("BTC pirce")
    ax.plot(time, df['BTC_Close'],label="real price")
    ax.plot(time, predict, label="predict price")
    plt.xlabel("year")
    plt.ylabel("price(USD)")
    plt.legend()
    ax.xaxis.set_major_formatter(md.DateFormatter('%Y'))
    plt.show()

