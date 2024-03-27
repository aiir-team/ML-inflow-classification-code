#!/usr/bin/env python
# Created by "Thieu" at 13:00, 13/09/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
# for chapter 6.3   https://michael-fuchs-python.netlify.app/2019/10/31/introduction-to-logistic-regression/#model-evaluation
from permetrics import RegressionMetric
from permetrics.classification import ClassificationMetric
from sklearn.metrics import roc_auc_score
from config import Config
import numpy as np


# def classify_metrics(y_true, y_pred, y_true_scaled=None, y_pred_scaled=None,
#                      n_labels=2, name_labels=None, positive_label=None, n_decimal=4):
#     accuracy = round(accuracy_score(y_true, y_pred), n_decimal)
#     error_rate = round(1 - accuracy_score(y_true, y_pred), n_decimal)
#     if n_labels == 2:
#         precision = round(precision_score(y_true, y_pred, pos_label=positive_label), n_decimal)
#         recall = round(recall_score(y_true, y_pred, pos_label=positive_label), n_decimal)
#         f1score = round(f1_score(y_true, y_pred, pos_label=positive_label), n_decimal)
#         auc = round(roc_auc_score(y_true, y_pred_scaled), n_decimal)
#     else:
#         precision = round(precision_score(y_true, y_pred, average='micro'), n_decimal)
#         recall = round(recall_score(y_true, y_pred, average='micro'), n_decimal)
#         f1score = round(f1_score(y_true, y_pred, average='micro'), n_decimal)
#         auc = round(roc_auc_score(y_true, y_pred_scaled, multi_class="ovr"), n_decimal)
#
#     ## Calculate metrics
#     metric_normal = {
#         "accuracy": accuracy,
#         "error": error_rate,
#         "precision": precision,
#         "recall": recall,
#         "f1": f1score,
#         "roc_auc": auc
#     }
#
#     # Confusion matrix
#     matrix_conf = confusion_matrix(y_true, y_pred, labels=name_labels)
#
#     # For figure purpose only
#     # logit_roc_auc = roc_auc_score(y_true, y_pred)
#     # fpr, tpr, thresholds = roc_curve(y_true, y_pred)
#
#     return matrix_conf, metric_normal


def class_metrics(y_true, y_pred, y_true_scaled=None, y_pred_scaled=None, n_labels=2, labels=None, n_decimal=5):
    evaluator = ClassificationMetric(y_true, y_pred, decimal=n_decimal)
    if n_labels == 2:
        paras = [{"average": "micro"}, ] * len(Config.METRICS_CLS_FOR_TESTING)
        metrics = evaluator.get_metrics_by_list_names(Config.METRICS_CLS_FOR_TESTING, paras)
        # metrics["ROC_AUC"] = round(roc_auc_score(y_true, y_pred_scaled), n_decimal)
    else:
        paras = [{"average": "macro"}, ] * len(Config.METRICS_CLS_FOR_TESTING)
        metrics = evaluator.get_metrics_by_list_names(Config.METRICS_CLS_FOR_TESTING, paras)
        # metrics["ROC_AUC"] = round(roc_auc_score(y_true, y_pred_scaled, multi_class="ovr"), n_decimal)
    # Confusion matrix

    matrix_conf = evaluator.confusion_matrix(y_true, y_pred, labels)

    # For figure purpose only
    # logit_roc_auc = roc_auc_score(y_true, y_pred)
    # fpr, tpr, thresholds = roc_curve(y_true, y_pred)

    return matrix_conf, metrics


def my_classify_metrics(y_true, y_pred, y_pred_prob=None, prefix="train"):
    evaluator = ClassificationMetric(y_true, y_pred)
    paras = [{"average": "weighted"}, ] * len(Config.METRICS_CLS_FOR_TESTING)
    mm2 = evaluator.get_metrics_by_list_names(Config.METRICS_CLS_FOR_TESTING, paras)
    if y_pred_prob is not None:
        y_pred_prob = y_pred_prob.astype(np.float_)
        if not np.allclose(1, y_pred_prob.sum(axis=1)):
            t1 = np.sum(y_pred_prob, axis=1).reshape((-1, 1))
            t1 = t1.repeat(y_pred_prob.shape[1], 1)
            y_pred_prob = np.divide(y_pred_prob, t1)
        mm2["ROC_AUC"] = round(roc_auc_score(y_true, y_pred_prob, multi_class="ovr"), 4)
    mm = {}
    for key, value in mm2.items():
        mm[f"{prefix}_{key}"] = value
    matrix_conf = evaluator.confusion_matrix(y_true, y_pred, labels=None)
    return matrix_conf, mm


def my_regression_metrics(y_true, y_pred, prefix="train"):
    evaluator = RegressionMetric(y_true, y_pred)
    mm2 = evaluator.get_metrics_by_list_names(Config.METRICS_REG_FOR_TESTING)
    mm = {}
    for key, value in mm2.items():
        mm[f"{prefix}_{key}"] = value
    return mm
