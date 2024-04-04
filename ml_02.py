#!/usr/bin/env python
# Created by "Thieu" at 23:08, 27/03/2024 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import LabelEncoder
from utils.data_util import convert_to_classification, divide_dataset_regression, scale_dataset_regression
from utils.result_util import save_classification_results, save_regression_results
from config import Config, Const
from utils.feature_util import select_features


data = pd.read_csv("data/input_data/inflow_by_mean.csv")
X = data[['value-1', 'value-2', 'value-3', 'value-4', 'value-5', 'value-6', 'value-7',
          'value-8', 'value-9', 'value-10', 'value-11', 'value-12']].values
y = data[["value", "month"]].values

## Divide dataset
x_train, x_test, y_train, y_test, index_test = divide_dataset_regression(X, y[:,0], test_size=Config.TEST_SIZE)

## Select features
key_features = "SVMSelector"
selected_features_idx, selected_features_score = select_features(x_train, y_train, mi_weight=0, anova_weight=0, dt_weight=0, rf_weight=0, svm_weight=1)
X_train = x_train[:, selected_features_idx[selected_features_score>Config.FS_THRESHOLD]]
X_test = x_test[:, selected_features_idx[selected_features_score>Config.FS_THRESHOLD]]
print(selected_features_idx)
print(selected_features_score)

## Scale dataset
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, scaler_y = scale_dataset_regression(X_train, X_test, y_train, y_test, scaler="std")

## Build models
list_models = [
    {
        "name": "RF",
        "model": RandomForestRegressor(random_state=Config.SEED),
        "param_grid": Config.SVM_RF_GRID
    }, {
        "name": "SVM",
        "model": SVR(),
        "param_grid": Config.SVM_SVM_GRID
    }, {
        "name": "LR",
        "model": LinearRegression(),
        "param_grid": Config.SVM_LR_GRID
    }, {
        "name": "KNN",
        "model": KNeighborsRegressor(),
        "param_grid": Config.SVM_KNN_GRID
    }, {
        "name": "DT",
        "model": DecisionTreeRegressor(random_state=Config.SEED),
        "param_grid": Config.SVM_DT_GRID
    }, {
        "name": "AdaBoost",
        "model": AdaBoostRegressor(random_state=Config.SEED),
        "param_grid": Config.SVM_AdaBoost_GRID
    }, {
        "name": "MLP",
        "model": MLPRegressor(random_state=Config.SEED),
        "param_grid": Config.SVM_MLP_GRID
    }
]


for idx_model, model in enumerate(list_models):
    grid = GridSearchCV(model['model'], model['param_grid'], refit=True, verbose=0, n_jobs=8, scoring="neg_mean_squared_error")
    grid.fit(X_train_scaled, y_train_scaled.ravel())
    mm0 = {
        "features": key_features,
        "model": model['name'],
        "best_params": grid.best_params_,
        "best_estimator": grid.best_estimator_
    }
    y_train_pred = grid.predict(X_train_scaled)
    y_test_pred = grid.predict(X_test_scaled)

    results_reg = {
        Const.Y_TRAIN_TRUE_SCALED: y_train_scaled,
        Const.Y_TRAIN_TRUE_UNSCALED: y_train,
        Const.Y_TRAIN_PRED_SCALED: y_train_pred,
        Const.Y_TRAIN_PRED_UNSCALED: scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)),

        Const.Y_TEST_TRUE_SCALED: y_test_scaled,
        Const.Y_TEST_TRUE_UNSCALED: y_test,
        Const.Y_TEST_PRED_SCALED: y_test_pred,
        Const.Y_TEST_PRED_UNSCALED: scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)),
    }
    save_regression_results(results=results_reg, validation=Config.VALIDATION_USED, metrics_head=mm0, metrics_file="metrics-reg-results",
                 test_filename=f"{model['name']}", pathsave=f"{Config.DATA_RESULTS_ALL}/{key_features}", loss_train=None)

    lb_encoder = LabelEncoder()
    y_train_true = convert_to_classification(y_train, month=y[:index_test, 1], matrix="mean")
    y_train_true_scaled = lb_encoder.fit_transform(y_train_true)
    y_test_true = convert_to_classification(y_test, month=y[index_test:, 1], matrix="mean")
    y_test_true_scaled = lb_encoder.transform(y_test_true)

    y_train_pred = convert_to_classification(scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)), month=y[:index_test, 1], matrix="mean")
    y_test_pred = convert_to_classification(scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)), month=y[index_test:, 1], matrix="mean")
    y_train_pred_scaled = lb_encoder.transform(y_train_pred)
    y_test_pred_scaled = lb_encoder.transform(y_test_pred)

    results_cls = {
        Const.Y_TRAIN_TRUE_SCALED: y_train_true_scaled,  # 0, 1, 2, 4
        Const.Y_TRAIN_TRUE_UNSCALED: y_train_true,  # categorical string
        Const.Y_TRAIN_PRED_SCALED: y_train_pred_scaled,  # 0 and 1
        Const.Y_TRAIN_PRED_UNSCALED: y_train_pred,  # categorical string

        Const.Y_TEST_TRUE_SCALED: y_test_true_scaled,
        Const.Y_TEST_TRUE_UNSCALED: y_test_true,
        Const.Y_TEST_PRED_SCALED: y_test_pred_scaled,
        Const.Y_TEST_PRED_UNSCALED: y_test_pred,

        Const.Y_TRAIN_PRED_PROB: None,
        Const.Y_TEST_PRED_PROB: None,
    }
    save_classification_results(results=results_cls, validation=Config.VALIDATION_USED, metrics_head=mm0, metrics_file="metrics-cls-results",
                 test_filename=f"{model['name']}",
                 pathsave=f"{Config.DATA_RESULTS_ALL}/{key_features}",
                 name_labels=lb_encoder.classes_, name_model=model['name'], n_labels=len(lb_encoder.classes_),
                 loss_train=None, system=None, verbose=False, draw_auc=False)
