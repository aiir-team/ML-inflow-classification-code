#!/usr/bin/env python
# Created by "Thieu" at 22:49, 29/03/2024 ----------%                                                                               
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
from utils.feature_util import select_reg_features, select_cls_features

import pandas as pd
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import LabelEncoder
from utils.data_util import divide_dataset_classification, scale_dataset_classification
from utils.result_util import save_classification_results
from config import Config, Const
from utils.feature_util import select_cls_features


def check_regression_features():
    data = pd.read_csv("data/input_data/inflow_by_mean.csv")
    X = data[['value-1', 'value-2', 'value-3', 'value-4', 'value-5', 'value-6', 'value-7',
              'value-8', 'value-9', 'value-10', 'value-11', 'value-12']].values
    y = data[["value", "month"]].values

    ## Divide dataset
    x_train, x_test, y_train, y_test, index_test = divide_dataset_regression(X, y[:,0], test_size=Config.TEST_SIZE)

    ## Select features
    selected_features_idx, selected_features_score = select_reg_features(x_train, y_train, mi_weight=1, anova_weight=0, dt_weight=0, rf_weight=0, svm_weight=0)
    print(selected_features_idx)
    print(selected_features_score)
    print(selected_features_idx[selected_features_score>Config.FS_REG_THRESHOLD])


    selected_features_idx, selected_features_score = select_reg_features(x_train, y_train, mi_weight=0, anova_weight=0, dt_weight=0, rf_weight=1, svm_weight=0)
    print(selected_features_idx)
    print(selected_features_score)
    print(selected_features_idx[selected_features_score>Config.FS_REG_THRESHOLD])


    selected_features_idx, selected_features_score = select_reg_features(x_train, y_train, mi_weight=0, anova_weight=0, dt_weight=0, rf_weight=0, svm_weight=1)
    print(selected_features_idx)
    print(selected_features_score)
    print(selected_features_idx[selected_features_score>Config.FS_REG_THRESHOLD])


    selected_features_idx, selected_features_score = select_reg_features(x_train, y_train, mi_weight=0.33, anova_weight=0, dt_weight=0, rf_weight=0.33, svm_weight=0.33)
    print(selected_features_idx)
    print(selected_features_score)
    print(selected_features_idx[selected_features_score>Config.FS_REG_THRESHOLD])


def check_classification_features():
    data = pd.read_csv("data/input_data/inflow_by_mean.csv")
    X = data[['value-1', 'value-2', 'value-3', 'value-4', 'value-5', 'value-6', 'value-7',
              'value-8', 'value-9', 'value-10', 'value-11', 'value-12']].values
    y = data[["label", "month"]].values
    lb_encoder = LabelEncoder()
    y_out = lb_encoder.fit_transform(y[:, 0])

    ## Divide dataset
    x_train, x_test, y_train, y_test, index_test = divide_dataset_regression(X, y_out, test_size=Config.TEST_SIZE)

    ## Scale dataset
    x_train, x_test, y_train, y_test, scaler_X, _ = scale_dataset_classification(x_train, x_test, y_train, y_test, scaler="std")

    ## Select features
    selected_features_idx, selected_features_score = select_cls_features(x_train, y_train, mi_weight=1, anova_weight=0, dt_weight=0, rf_weight=0, svm_weight=0)
    print(selected_features_idx)
    print(selected_features_score)
    print(selected_features_idx[selected_features_score > Config.FS_CLS_THRESHOLD])

    selected_features_idx, selected_features_score = select_cls_features(x_train, y_train, mi_weight=0, anova_weight=0, dt_weight=0, rf_weight=0, svm_weight=1)
    print(selected_features_idx)
    print(selected_features_score)
    print(selected_features_idx[selected_features_score > Config.FS_CLS_THRESHOLD])

    selected_features_idx, selected_features_score = select_cls_features(x_train, y_train, mi_weight=0, anova_weight=0, dt_weight=0, rf_weight=1, svm_weight=0)
    print(selected_features_idx)
    print(selected_features_score)
    print(selected_features_idx[selected_features_score > Config.FS_CLS_THRESHOLD])

    selected_features_idx, selected_features_score = select_cls_features(x_train, y_train, mi_weight=0.33, anova_weight=0, dt_weight=0, rf_weight=0.33, svm_weight=0.33)
    print(selected_features_idx)
    print(selected_features_score)
    print(selected_features_idx[selected_features_score > Config.FS_CLS_THRESHOLD])


def check_classification_smote_features():
    data = pd.read_csv("data/input_data/inflow_by_mean.csv")
    X = data[['value-1', 'value-2', 'value-3', 'value-4', 'value-5', 'value-6', 'value-7',
              'value-8', 'value-9', 'value-10', 'value-11', 'value-12']].values
    y_raw = data[["label", "month"]].values
    lb_encoder = LabelEncoder()
    y_clean = lb_encoder.fit_transform(y_raw[:, 0])

    ## Divide dataset
    x_train, x_test, y_train, y_test = divide_dataset_classification(X, y_clean, test_size=Config.TEST_SIZE)

    ## Scale dataset
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, scaler_X, _ = scale_dataset_classification(x_train, x_test, y_train, y_test,
                                                                                                             scaler="std", fix_imbalanced=True)

    ## Select features
    selected_features_idx, selected_features_score = select_cls_features(X_train_scaled, y_train_scaled, mi_weight=1, anova_weight=0, dt_weight=0, rf_weight=0, svm_weight=0)
    print(selected_features_idx)
    print(selected_features_score)
    print(selected_features_idx[selected_features_score > Config.FS_CLS_THRESHOLD_SMOTE])

    selected_features_idx, selected_features_score = select_cls_features(x_train, y_train, mi_weight=0, anova_weight=0, dt_weight=0, rf_weight=0, svm_weight=1)
    print(selected_features_idx)
    print(selected_features_score)
    print(selected_features_idx[selected_features_score > Config.FS_CLS_THRESHOLD_SMOTE])

    selected_features_idx, selected_features_score = select_cls_features(x_train, y_train, mi_weight=0, anova_weight=0, dt_weight=0, rf_weight=1, svm_weight=0)
    print(selected_features_idx)
    print(selected_features_score)
    print(selected_features_idx[selected_features_score > Config.FS_CLS_THRESHOLD_SMOTE])

    selected_features_idx, selected_features_score = select_cls_features(x_train, y_train, mi_weight=0.33, anova_weight=0, dt_weight=0, rf_weight=0.33, svm_weight=0.33)
    print(selected_features_idx)
    print(selected_features_score)
    print(selected_features_idx[selected_features_score > Config.FS_CLS_THRESHOLD_SMOTE])




# check_regression_features()
check_classification_features()
# check_classification_smote_features()
