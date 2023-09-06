#!/usr/bin/env python
# Created by "Thieu" at 13:53, 06/09/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import pandas as pd
from xgboost import XGBClassifier
from config import Config
from utils.data_util import split_dataset
from utils.math_util import get_combinations
from utils.io_util import save_results_to_csv
from permetrics.classification import ClassificationMetric
import warnings

warnings.filterwarnings('ignore')


# Load Train and Test datasets
# Identify feature and response variable(s) and values must be numeric and numpy arrays
data = pd.read_csv('data/input_data/inflow_by_mean.csv')
list_features = ['value', 'value-1', 'value-2', 'value-3', 'value-4', 'value-5',
                 'value-6', 'value-7', 'value-8', 'value-9', 'value-10', 'value-11', 'value-12']
list_all_features = get_combinations(list_features)
y_output = 'label+1'


for idx, features in enumerate(list_all_features):
    if len(features) <= 3:
        continue
    x_train, x_test, y_train, y_test, lb_encoder = split_dataset(data, features, y_output)

    model = XGBClassifier().fit(x_train, y_train)

    mm1 = {
        "features": features,
        "best_params": "None",
        "best_estimator": "XGB"
    }
    predicted = model.predict(x_test)
    evaluator = ClassificationMetric(y_test, predicted, decimal=4)
    metrics = ["AS", "PS", "RS", "F1S", "F2S", "MCC", "LS"]
    paras = [{"average": "micro"}, ] * len(metrics)
    mm2 = evaluator.get_metrics_by_list_names(metrics, paras)
    mm = {**mm1, **mm2}
    save_results_to_csv(mm, f"xgb-tuning-results_mean_threshold", "data/history")
