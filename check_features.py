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
from utils.feature_util import select_reg_features


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
print(selected_features_idx[selected_features_score>Config.FS_THRESHOLD])


selected_features_idx, selected_features_score = select_reg_features(x_train, y_train, mi_weight=0, anova_weight=0, dt_weight=0, rf_weight=1, svm_weight=0)
print(selected_features_idx)
print(selected_features_score)
print(selected_features_idx[selected_features_score>Config.FS_THRESHOLD])


selected_features_idx, selected_features_score = select_reg_features(x_train, y_train, mi_weight=0, anova_weight=0, dt_weight=0, rf_weight=0, svm_weight=1)
print(selected_features_idx)
print(selected_features_score)
print(selected_features_idx[selected_features_score>Config.FS_THRESHOLD])


selected_features_idx, selected_features_score = select_reg_features(x_train, y_train, mi_weight=0.33, anova_weight=0, dt_weight=0, rf_weight=0.33, svm_weight=0.33)
print(selected_features_idx)
print(selected_features_score)
print(selected_features_idx[selected_features_score>Config.FS_THRESHOLD])
