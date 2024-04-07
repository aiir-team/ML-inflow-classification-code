#!/usr/bin/env python
# Created by "Thieu" at 16:39, 04/04/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

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

data = pd.read_csv("data/input_data/inflow_by_mean.csv")
X = data[['value-1', 'value-2', 'value-3', 'value-4', 'value-5', 'value-6', 'value-7',
          'value-8', 'value-9', 'value-10', 'value-11', 'value-12']].values
y_raw = data[["label", "month"]].values
lb_encoder = LabelEncoder()
y_clean = lb_encoder.fit_transform(y_raw[:, 0])

## Divide dataset
x_train, x_test, y_train, y_test = divide_dataset_classification(X, y_clean, test_size=Config.TEST_SIZE)

## Scale dataset
X_train_scaled, X_test_scaled, y_train, y_test, scaler_X, _ = scale_dataset_classification(x_train, x_test, y_train, y_test, scaler="std")

## Select features
key_features = "MCFSSelector"
selected_features_idx, selected_features_score = select_cls_features(X_train_scaled, y_train, mi_weight=0.33, anova_weight=0, dt_weight=0, rf_weight=0.33, svm_weight=0.33)
X_train_scaled = X_train_scaled[:, selected_features_idx[selected_features_score>Config.FS_CLS_THRESHOLD]]
X_test_scaled = X_test_scaled[:, selected_features_idx[selected_features_score>Config.FS_CLS_THRESHOLD]]
print(selected_features_idx)
print(selected_features_score)

## Build models
list_models = [
    {
        "name": "RF",
        "model": RandomForestClassifier(random_state=Config.SEED),
        "param_grid": Config.MCFS_RF_GRID_CLS
    }, {
        "name": "SVM",
        "model": SVC(probability=True, random_state=Config.SEED),
        "param_grid": Config.MCFS_SVM_GRID_CLS
    }, {
        "name": "LR",
        "model": LogisticRegression(random_state=Config.SEED),
        "param_grid": Config.MCFS_LR_GRID_CLS
    }, {
        "name": "KNN",
        "model": KNeighborsClassifier(),
        "param_grid": Config.MCFS_KNN_GRID_CLS
    }, {
        "name": "DT",
        "model": DecisionTreeClassifier(random_state=Config.SEED),
        "param_grid": Config.MCFS_DT_GRID_CLS
    }, {
        "name": "AdaBoost",
        "model": AdaBoostClassifier(random_state=Config.SEED),
        "param_grid": Config.MCFS_AdaBoost_GRID_CLS
    }, {
        "name": "MLP",
        "model": MLPClassifier(random_state=Config.SEED),
        "param_grid": Config.MCFS_MLP_GRID_CLS
    }
]

for idx_model, model in enumerate(list_models):
    grid = GridSearchCV(model['model'], model['param_grid'], refit=True, verbose=0, n_jobs=8, scoring="f1_macro")
    grid.fit(X_train_scaled, y_train)
    mm0 = {
        "features": key_features,
        "model": model['name'],
        "best_params": grid.best_params_,
        "best_estimator": grid.best_estimator_
    }
    y_train_pred = grid.predict(X_train_scaled)
    y_test_pred = grid.predict(X_test_scaled)
    results = {
        Const.Y_TRAIN_TRUE_SCALED: y_train,  # 0 and 1
        Const.Y_TRAIN_TRUE_UNSCALED: lb_encoder.inverse_transform(y_train),  # categorical string
        Const.Y_TRAIN_PRED_SCALED: y_train_pred,  # 0 and 1
        Const.Y_TRAIN_PRED_UNSCALED: lb_encoder.inverse_transform(y_train_pred),  # categorical string

        Const.Y_TEST_TRUE_SCALED: y_test,
        Const.Y_TEST_TRUE_UNSCALED: lb_encoder.inverse_transform(y_test),
        Const.Y_TEST_PRED_SCALED: y_test_pred,
        Const.Y_TEST_PRED_UNSCALED: lb_encoder.inverse_transform(y_test_pred),

        Const.Y_TRAIN_PRED_PROB: grid.predict_proba(X_train_scaled),
        Const.Y_TEST_PRED_PROB: grid.predict_proba(X_test_scaled),
    }
    save_classification_results(results=results, validation=Config.VALIDATION_USED, metrics_head=mm0, metrics_file="metrics-results-cls",
                                test_filename=f"{model['name']}",
                                pathsave=f"{Config.DATA_RESULTS_CLS}/{key_features}",
                                name_labels=lb_encoder.classes_, name_model=model['name'], n_labels=len(lb_encoder.classes_),
                                loss_train=None, system=None, verbose=False, draw_auc=True)
