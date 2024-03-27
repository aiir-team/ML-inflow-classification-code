#!/usr/bin/env python
# Created by "Thieu" at 14:30, 25/10/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from utils.data_util import split_smote_dataset
from utils.io_util import save_results_to_csv
from utils.math_util import get_combinations
from utils.metric_util import my_classify_metrics
import warnings
warnings.filterwarnings('ignore')

# Identify feature and response variable(s) and values must be numeric and numpy arrays
data = pd.read_csv('data/input_data/inflow_by_mean.csv')
list_features = ['value', 'value-1', 'value-2', 'value-3', 'value-4', 'value-5',
                 'value-6', 'value-7', 'value-8', 'value-9', 'value-10', 'value-11', 'value-12']
list_all_features = get_combinations(list_features)
y_output = 'label+1'


for idx, features in enumerate(list_all_features):
    if len(features) <= 3:
        continue
    x_train, x_test, y_train, y_test, lb_encoder = split_smote_dataset(data, features, y_output)
    # defining parameter range
    param_grid = {'C': [0.1, 1, 10, 100, 1000],
                  'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
                  'kernel': ['rbf', 'linear']}

    grid = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3)
    grid.fit(x_train, y_train)

    mm0 = {
        "features": features,
        "best_params": grid.best_params_,
        "best_estimator": grid.best_estimator_
    }
    y_test_pred = grid.predict(x_test)
    y_train_pred = grid.predict(x_train)
    y_test_pred_prob = grid.predict_proba(x_test)
    y_train_pred_prob = grid.predict_proba(x_train)
    mm1 = my_classify_metrics(y_train, y_train_pred, y_train_pred_prob, prefix="train")
    mm2 = my_classify_metrics(y_test, y_test_pred, y_test_pred_prob, prefix="test")
    mm = {**mm0, **mm1, **mm2}
    save_results_to_csv(mm, f"smote-svm-tuning-results_mean_threshold", "data/history")
