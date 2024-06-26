#!/usr/bin/env python
# Created by "Thieu" at 16:51, 26/03/2024 ----------%                                                                               
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
from utils.data_util import split_dataset_regression, convert_to_classification
from utils.result_util import save_classification_results, save_regression_results
from config import Config, Const


def get_feature_by_DT_selector(X=None, y=None):
    # Train Random Forest
    regressor = DecisionTreeRegressor()
    regressor.fit(X, y)

    # Get feature importance
    feature_importances = regressor.feature_importances_

    # Print feature importance ranking
    print("Feature Importance Ranking:")
    for i, importance in enumerate(feature_importances):
        print(f"Feature {i}: Importance {importance}")

    # Select features based on importance (for example, select top 4 features)
    top_feature_indices = feature_importances.argsort()[-4:][::-1]
    X_new = X[:, top_feature_indices]
    return X_new


data = pd.read_csv("data/input_data/inflow_by_mean.csv")
X = data[['value-1', 'value-2', 'value-3', 'value-4', 'value-5', 'value-6', 'value-7',
          'value-8', 'value-9', 'value-10', 'value-11', 'value-12']].values
y = data[["value", "month"]].values

X_new = get_feature_by_DT_selector(X, y[:,0])

## Split train and test
x_train, x_test, y_train, y_test, scaler_X, scaler_y, index_test = split_dataset_regression(X_new, y[:,0], scaler="std")


list_models = [
    {
        "name": "RF",
        "model": RandomForestRegressor(),
        "param_grid": {'n_estimators': [10, 20, 30, 40, 50]}
    }, {
        "name": "SVM",
        "model": SVR(),
        "param_grid": {'C': [0.1, 1., 5., 10., 15.],
                  'gamma': ['scale', 'auto'],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    }, {
        "name": "LR",
        "model": LinearRegression(),
        "param_grid": {'fit_intercept': [True, False]}
    }, {
        "name": "KNN",
        "model": KNeighborsRegressor(),
        "param_grid": {'n_neighbors': [5, 10, 15, 20, 25]}
    }, {
        "name": "DT",
        "model": DecisionTreeRegressor(),
        "param_grid": {'criterion': ["squared_error", "absolute_error", "poisson"]}
    }, {
        "name": "AdaBoost",
        "model": AdaBoostRegressor(),
        "param_grid": {'n_estimators': [10, 20, 30, 40, 50],
                       'loss': ['linear', 'square', 'exponential']}
    }, {
        "name": "MLP",
        "model": MLPRegressor(),
        "param_grid": {'hidden_layer_sizes': list(range(5, 55, 5)),
                       'activation': ['logistic', 'tanh', 'relu'],
                       'solver': ['lbfgs', 'sgd', 'adam'],
                       'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                       'max_iter': list(range(1000, 2100, 100))}
    }
]


key_features = "RfSelector"
for idx_model, model in enumerate(list_models):
    grid = GridSearchCV(model['model'], model['param_grid'], refit=True, verbose=0, n_jobs=8, scoring="neg_mean_squared_error")
    grid.fit(x_train, y_train.ravel())
    mm0 = {
        "features": key_features,
        "model": model['name'],
        "best_params": grid.best_params_,
        "best_estimator": grid.best_estimator_
    }
    y_train_pred = grid.predict(x_train)
    y_test_pred = grid.predict(x_test)

    results_reg = {
        Const.Y_TRAIN_TRUE_SCALED: y_train,
        Const.Y_TRAIN_TRUE_UNSCALED: scaler_y.inverse_transform(y_train),
        Const.Y_TRAIN_PRED_SCALED: y_train_pred,
        Const.Y_TRAIN_PRED_UNSCALED: scaler_y.inverse_transform(y_train_pred.reshape(-1, 1)),

        Const.Y_TEST_TRUE_SCALED: y_test,
        Const.Y_TEST_TRUE_UNSCALED: scaler_y.inverse_transform(y_test),
        Const.Y_TEST_PRED_SCALED: y_test_pred,
        Const.Y_TEST_PRED_UNSCALED: scaler_y.inverse_transform(y_test_pred.reshape(-1, 1)),
    }
    save_regression_results(results=results_reg, validation=Config.VALIDATION_USED, metrics_head=mm0, metrics_file="metrics-reg-results",
                 test_filename=f"{key_features}-{model['name']}", pathsave=f"{Config.DATA_RESULTS_DT}", loss_train=None)

    lb_encoder = LabelEncoder()
    y_train_true = convert_to_classification(scaler_y.inverse_transform(y_train), month=y[:index_test, 1], matrix="mean")
    y_train_true_scaled = lb_encoder.fit_transform(y_train_true)
    y_test_true = convert_to_classification(scaler_y.inverse_transform(y_test), month=y[index_test:, 1], matrix="mean")
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
                 test_filename=f"{key_features}-{model['name']}",
                 pathsave=f"{Config.DATA_RESULTS_DT}",
                 name_labels=lb_encoder.classes_, name_model=model['name'], n_labels=len(lb_encoder.classes_),
                 loss_train=None, system=None, verbose=False, draw_auc=False)
