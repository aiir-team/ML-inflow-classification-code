#!/usr/bin/env python
# Created by "Thieu" at 23:00, 27/03/2024 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from sklearn.feature_selection import mutual_info_regression, f_regression, mutual_info_classif, f_classif
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
# from sklearn.datasets import load_iris, load_boston


def select_reg_features(X, y, mi_weight=0.2, anova_weight=0.2, dt_weight=0.2, rf_weight=0.2, svm_weight=0.2):
    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y, random_state=42)
    # Perform ANOVA test and get F-values
    f_values, _ = f_regression(X, y)

    # Fit decision tree and random forest models
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X, y)
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X, y)

    # Fit Support Vector Machine (SVM) model
    svm = SVR(kernel='linear')
    svm.fit(X, y)
    # Get SVM weights
    svm_weights = np.abs(svm.coef_)
    # Normalize SVM weights
    svm_weights_normalized = (svm_weights - np.min(svm_weights)) / (np.max(svm_weights) - np.min(svm_weights))

    # Normalize mutual information scores
    mi_scores_normalized = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores))
    # Normalize F-values
    f_values_normalized = (f_values - np.min(f_values)) / (np.max(f_values) - np.min(f_values))
    # Normalize decision tree feature importances
    dt_importances_normalized = (dt.feature_importances_ - np.min(dt.feature_importances_)) / (
                np.max(dt.feature_importances_) - np.min(dt.feature_importances_))
    # Normalize random forest feature importances
    rf_importances_normalized = (rf.feature_importances_ - np.min(rf.feature_importances_)) / (
                np.max(rf.feature_importances_) - np.min(rf.feature_importances_))

    # Calculate weighted scores
    combined_scores = (mi_weight * mi_scores_normalized +
                       anova_weight * f_values_normalized +
                       dt_weight * dt_importances_normalized +
                       rf_weight * rf_importances_normalized +
                       svm_weight * svm_weights_normalized).ravel()

    # Sort features based on their combined scores in descending order
    sorted_features_indices = np.argsort(combined_scores)[::-1]
    sorted_scores = combined_scores[sorted_features_indices]

    return sorted_features_indices, sorted_scores


#     # # Create a dictionary to store feature indices and their combined scores
#     # feature_scores = dict(zip(range(X.shape[1]), combined_scores.ravel()))
#     #
#     # # Sort features based on their combined scores in descending order
#     # sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
#     #
#     # return sorted_features
#
#
# # Example usage:
# data = load_boston()
# X = data.data
# y = data.target
#
# # selected_features = select_reg_features(X, y, mi_weight=0.2, anova_weight=0.2, dt_weight=0.2, rf_weight=0.2, svm_weight=0.2)
# # # Print the selected features
# # print("Selected Features:")
# # for i, (feature_idx, score) in enumerate(selected_features, start=1):
# #     feature_name = data.feature_names[feature_idx]
# #     print(f"{i}. {feature_name}: Combined Score = {score:.4f}")
#
#
# selected_features_indices, selected_scores = select_reg_features(X, y, mi_weight=0.2, anova_weight=0.2, dt_weight=0.2, rf_weight=0.2, svm_weight=0.2)
#
# # Print the selected features
# print("Selected Features:")
# for i, (feature_idx, score) in enumerate(zip(selected_features_indices, selected_scores), start=1):
#     feature_name = data.feature_names[feature_idx]
#     print(f"{i}. {feature_idx}, {feature_name}: Combined Score = {score:.4f}")


def select_cls_features(X, y, mi_weight=0.2, anova_weight=0.2, dt_weight=0.2, rf_weight=0.2, svm_weight=0.2):
    # Calculate mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    # Perform ANOVA test and get F-values
    f_values, _ = f_classif(X, y)

    # Fit decision tree and random forest models
    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X, y)
    rf = RandomForestClassifier(n_estimators=30, random_state=42)
    rf.fit(X, y)

    # Fit Support Vector Machine (SVM) model
    svm = SVC(kernel='linear', probability=True, random_state=42)
    svm.fit(X, y)
    # Get SVM weights
    svm_weights = np.abs(svm.coef_[0])

    # Normalize scores
    mi_scores_normalized = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores))
    f_values_normalized = (f_values - np.min(f_values)) / (np.max(f_values) - np.min(f_values))
    dt_importances_normalized = (dt.feature_importances_ - np.min(dt.feature_importances_)) / (
                np.max(dt.feature_importances_) - np.min(dt.feature_importances_))
    rf_importances_normalized = (rf.feature_importances_ - np.min(rf.feature_importances_)) / (
                np.max(rf.feature_importances_) - np.min(rf.feature_importances_))
    svm_weights_normalized = (svm_weights - np.min(svm_weights)) / (np.max(svm_weights) - np.min(svm_weights))

    # Calculate combined scores
    combined_scores = (mi_weight * mi_scores_normalized +
                       anova_weight * f_values_normalized +
                       dt_weight * dt_importances_normalized +
                       rf_weight * rf_importances_normalized +
                       svm_weight * svm_weights_normalized)

    # Sort features based on their combined scores in descending order
    sorted_features_indices = np.argsort(combined_scores)[::-1]
    sorted_scores = combined_scores[sorted_features_indices]

    return sorted_features_indices, sorted_scores
