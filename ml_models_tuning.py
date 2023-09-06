#!/usr/bin/env python
# Created by "Thieu" at 21:33, 01/11/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from config import Config, Const
from utils.data_util import split_dataset, split_smote_dataset
from utils.result_util import save_results
import warnings


warnings.filterwarnings('ignore')

# Identify feature and response variable(s) and values must be numeric and numpy arrays
data = pd.read_csv('data/input_data/inflow_by_mean.csv')
dict_features = {
    "ANOVA": ['value-2', 'value-3', 'value-4', 'value-5', 'value-6', 'value-10', 'value-11', 'value-12'],
    "MI": ['value', 'value-2', 'value-3', 'value-4', 'value-5', 'value-6', 'value-7', 'value-9', 'value-10', 'value-11', 'value-12']
}

y_output = 'label+1'

list_models = [
    {
        "name": "RF",
        "model": RandomForestClassifier(),
        "param_grid": {'n_estimators': list(range(5, 105, 5))}
    }, {
        "name": "SVM",
        "model": svm.SVC(probability=True),
        "param_grid": {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'linear']}
    }, {
        "name": "LR",
        "model": LogisticRegression(max_iter=1000),
        "param_grid": {'C': [0.01, 0.1, 1.0, 10, 100],
                       'max_iter': list(range(200, 2100, 100))}
    }, {
        "name": "KNN",
        "model": KNeighborsClassifier(),
        "param_grid": {'n_neighbors': list(range(3, 30))}
    }, {
        "name": "DT",
        "model": DecisionTreeClassifier(),
        "param_grid": {'criterion': ["gini", "entropy", "log_loss"],
                       'splitter': ['best', 'random']}
    }, {
        "name": "AdaBoost",
        "model": AdaBoostClassifier(),
        "param_grid": {'n_estimators': list(range(10, 205, 5)),
                       'algorithm': ['SAMME', 'SAMME.R']}
    }, {
        "name": "MLP",
        "model": MLPClassifier(),
        "param_grid": {'hidden_layer_sizes': list(range(10, 105, 5)),
                       'activation': ['logistic', 'tanh', 'relu'],
                       'solver': ['lbfgs', 'sgd', 'adam'],
                       'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                       'max_iter': list(range(500, 2100, 100))}
    }
]

list_models = [{
        "name": "MLP",
        "model": MLPClassifier(),
        "param_grid": {'hidden_layer_sizes': list(range(20, 105, 10)),
                       'activation': ['logistic', 'tanh', 'relu'],
                       'solver': ['lbfgs', 'sgd', 'adam'],
                       'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                       'max_iter': list(range(1000, 2100, 100))}
    }
]

list_data_handling = ["normal", "smote"]

# list_models = [
#     {
#         "name": "RF",
#         "model": RandomForestClassifier(),
#         "param_grid": {'n_estimators': [10, 20]}
#     }, {
#         "name": "SVM",
#         "model": svm.SVC(probability=True),
#         "param_grid": {'C': [0.1, 1],
#                   'gamma': [0.001],
#                   'kernel': ['rbf']}
#     }, {
#         "name": "LR",
#         "model": LogisticRegression(max_iter=1000),
#         "param_grid": {'C': [0.01, 0.1,]}
#     }, {
#         "name": "KNN",
#         "model": KNeighborsClassifier(),
#         "param_grid": {'n_neighbors': [5, 10]}
#     }, {
#         "name": "DT",
#         "model": DecisionTreeClassifier(),
#         "param_grid": {'criterion': ["gini", "entropy",],
#                        'splitter': ['best',]}
#     }, {
#         "name": "AdaBoost",
#         "model": AdaBoostClassifier(),
#         "param_grid": {'n_estimators': [10, 20],
#                        'algorithm': ['SAMME', 'SAMME.R']}
#     }, {
#         "name": "MLP",
#         "model": MLPClassifier(max_iter=1000),
#         "param_grid": {'hidden_layer_sizes': [20],
#                        'activation': ['tanh', 'relu'],
#                        'solver': ['adam'],
#                        'alpha': [0.0001, 0.001]}
#     }
# ]


for key_feature, features in dict_features.items():

    for idx_data, data_handling in enumerate(list_data_handling):
        if data_handling == "normal":
            x_train, x_test, y_train, y_test, scaler, lb_encoder = split_dataset(data, features, y_output)
        else:
            x_train, x_test, y_train, y_test, scaler, lb_encoder = split_smote_dataset(data, features, y_output)

        for idx_model, model in enumerate(list_models):
            grid = GridSearchCV(model['model'], model['param_grid'], refit=True, verbose=0, n_jobs=8)
            grid.fit(x_train, y_train)
            mm0 = {
                "features": key_feature,
                "model": model['name'],
                "best_params": grid.best_params_,
                "best_estimator": grid.best_estimator_
            }
            y_train_pred = grid.predict(x_train)
            y_test_pred = grid.predict(x_test)
            results = {
                Const.Y_TRAIN_TRUE_SCALED: y_train,                                   # 0 and 1
                Const.Y_TRAIN_TRUE_UNSCALED: lb_encoder.inverse_transform(y_train),       # categorical string
                Const.Y_TRAIN_PRED_SCALED: y_train_pred,                # 0 and 1
                Const.Y_TRAIN_PRED_UNSCALED: lb_encoder.inverse_transform(y_train_pred),                        # categorical string

                Const.Y_TEST_TRUE_SCALED: y_test,
                Const.Y_TEST_TRUE_UNSCALED: lb_encoder.inverse_transform(y_test),
                Const.Y_TEST_PRED_SCALED: y_test_pred,
                Const.Y_TEST_PRED_UNSCALED: lb_encoder.inverse_transform(y_test_pred),

                Const.Y_TRAIN_PRED_PROB: grid.predict_proba(x_train),
                Const.Y_TEST_PRED_PROB: grid.predict_proba(x_test),
            }
            save_results(results=results, validation=Config.VALIDATION_USED, metrics_head=mm0, metrics_file="metrics-results",
                         test_filename=f"{key_feature}-{model['name']}",
                         pathsave=f"{Config.DATA_RESULTS}/{data_handling}",
                         name_labels=lb_encoder.classes_, name_model=model['name'], n_labels=len(lb_encoder.classes_),
                         loss_train=None, system=None, verbose=False)

