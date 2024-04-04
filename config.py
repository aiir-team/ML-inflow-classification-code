#!/usr/bin/env python
# Created by "Thieu" at 23:27, 22/02/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from os.path import abspath, dirname

basedir = abspath(dirname(__file__))


VALIDATION_USED = False

if VALIDATION_USED:
    FILE_METRIC_CSV = ["AS_train", "PS_train", "RS_train", "F1S_train", "F2S_train", "MCC_train", "LS_train",
                       "AS_valid", "PS_valid", "RS_valid", "F1S_valid", "F2S_valid", "MCC_valid", "LS_valid",
                       "AS_test", "PS_test", "RS_test", "F1S_test", "F2S_test", "MCC_test", "LS_test"]
else:
    FILE_METRIC_CSV = ["AS_train", "PS_train", "RS_train", "F1S_train", "F2S_train", "MCC_train", "LS_train",
                       "AS_test", "PS_test", "RS_test", "F1S_test", "F2S_test", "MCC_test", "LS_test"]


class Const:

    Y_TRAIN_TRUE_SCALED = "y_train_true_scaled"
    Y_TRAIN_TRUE_UNSCALED = "y_train_true_unscaled"
    Y_TRAIN_PRED_SCALED = "y_train_pred_scaled"
    Y_TRAIN_PRED_UNSCALED = "y_train_pred_unscaled"

    Y_VALID_TRUE_SCALED = "y_valid_true_scaled"
    Y_VALID_TRUE_UNSCALED = "y_valid_true_unscaled"
    Y_VALID_PRED_SCALED = "y_valid_pred_scaled"
    Y_VALID_PRED_UNSCALED = "y_valid_pred_unscaled"

    Y_TEST_TRUE_SCALED = "y_test_true_scaled"
    Y_TEST_TRUE_UNSCALED = "y_test_true_unscaled"
    Y_TEST_PRED_SCALED = "y_test_pred_scaled"
    Y_TEST_PRED_UNSCALED = "y_test_pred_unscaled"

    Y_TRAIN_PRED_PROB = "y_train_pred_prob"
    Y_TEST_PRED_PROB = "y_test_pred_prob"
    Y_VALID_PRED_PROB = "y_valid_pred_prob"

    FILENAME_LOSS_TRAIN = "loss_train"
    
    FILENAME_PRED_TRAIN_CLS = "pred_train_cls"
    FILENAME_PRED_VALID_CLS = "pred_valid_cls"
    FILENAME_PRED_TEST_CLS = "pred_test_cls"
    FILENAME_PRED_REAL_WORLD_CLS = "pred_real_world_cls"

    FILENAME_PRED_TRAIN_REG = "pred_train_reg"
    FILENAME_PRED_VALID_REG = "pred_valid_reg"
    FILENAME_PRED_TEST_REG = "pred_test_reg"
    FILENAME_PRED_REAL_WORLD_REG = "pred_real_world_reg"

    FILENAME_IMG_CM_TRAIN = "confusion_matrix_train"
    FILENAME_IMG_CM_VALID = "confusion_matrix_valid"
    FILENAME_IMG_CM_TEST = "confusion_matrix_test"

    FILENAME_IMG_LOSS_ACC = "loss-accuracy"
    FILENAME_IMG_ROC_AUC = "roc_curve"

    FILENAME_IMG_PRE_RECALL = "precision_recall_curve"
    FILENAME_CONVERGENCE = "convergence"

    FILENAME_PERFORMANCE = "performance"
    FILENAME_METRICS = "metrics"
    FILENAME_MODEL = "model"
    FILENAME_METRICS_ALL_MODELS = "all-models-metrics"

    FILE_MIN = "min.csv"
    FILE_MEAN = "mean.csv"
    FILE_MAX = "max.csv"
    FILE_STD = "std.csv"
    FILE_CV = "cv.csv"
    FOLDERNAME_STATISTICS = "statistics"
    FOLDER_VISUALIZE = "visualize"

    HEADER_TRUTH_PREDICTED_TRAIN_FILE = [Y_TRAIN_TRUE_SCALED, Y_TRAIN_PRED_SCALED, Y_TRAIN_TRUE_UNSCALED, Y_TRAIN_PRED_UNSCALED]
    HEADER_TRUTH_PREDICTED_VALID_FILE = [Y_VALID_TRUE_SCALED, Y_VALID_PRED_SCALED, Y_VALID_TRUE_UNSCALED, Y_VALID_PRED_UNSCALED]
    HEADER_TRUTH_PREDICTED_TEST_FILE = [Y_TEST_TRUE_SCALED, Y_TEST_PRED_SCALED, Y_TEST_TRUE_UNSCALED, Y_TEST_PRED_UNSCALED]

    TITLE_IMG_CONF_MATRIX_TRAIN = "Confusion Matrix for training set"
    TITLE_IMG_CONF_MATRIX_VALID = "Confusion Matrix for validation set"
    TITLE_IMG_CONF_MATRIX_TEST = "Confusion Matrix for testing set"

    TITLE_IMG_ROC_AUC_MULTI = "Multiclass ROC curve"
    TITLE_IMG_PRE_RECALL_MULTI = "Multiclass Precision Recall curve"
    TITLE_IMG_ROC_AUC = "ROC curve"
    TITLE_IMG_PRE_RECALL = "Precision Recall curve"

    FILE_LOSS_HEADER = ["epoch", "loss", "val_loss"]
    FILE_FIGURE_TYPES = [".png", ".pdf"]    #[".png", ".pdf", ".eps"]

    TANH_LOSSES = ['hinge', 'squared_hinge']
    SIGMOID_LOSSES = ['binary_crossentropy']
    SOFTMAX_LOSSES = ['categorical_crossentropy', 'categorical_hinge', 'kl_divergence']


class Config:
    DATA_DIRECTORY = f'{basedir}/data'
    DATA_INPUT = f'{DATA_DIRECTORY}/input_data'
    DATA_RESULTS = f'{DATA_DIRECTORY}/results_01'

    DATA_RESULTS_RF = f'{DATA_DIRECTORY}/results_rf'
    DATA_RESULTS_DT = f'{DATA_DIRECTORY}/results_dt'
    DATA_RESULTS_ALL = f'{DATA_DIRECTORY}/results_final_02'

    FILE_METRIC_NAME = "metrics-results"
    FILE_METRIC_CSV_HEADER = ["model_paras", ] + FILE_METRIC_CSV
    FILE_METRIC_CSV_HEADER_FULL = ["path_paras", "trial", "model", "model_paras", ] + FILE_METRIC_CSV
    FILE_METRIC_CSV_HEADER_CALCULATE = FILE_METRIC_CSV
    FILE_METRIC_HEADER_STATISTICS = ["path_paras", "model", "model_paras"] + FILE_METRIC_CSV

    LEGEND_NETWORK = "Network = "
    LEGEND_EPOCH = "Number of Generations = "
    LEGEND_POP_SIZE = "Population size = "
    LEGEND_GROUNDTRUTH = "Ground Truth"
    LEGEND_PREDICTED = "Predicted"

    MHA_MODE_TRAIN_PHASE1 = "sequential"        # Don't change this value

    N_CPUS_RUN = 8
    SEED = 20
    VERBOSE = False
    N_TRIALS = 10                # Number of trials for each model
    FS_REG_THRESHOLD = 0.2
    FS_CLS_THRESHOLD = 0.35

    # Identify feature and response variable(s) and values must be numeric and numpy arrays
    NAME_DATASET = "data/input_data/inflow_by_mean.csv"
    DICT_FEATURES_X = {
        "ANOVA": ['value-2', 'value-3', 'value-4', 'value-5', 'value-6', 'value-10', 'value-11', 'value-12'],
        "MI": ['value', 'value-2', 'value-3', 'value-4', 'value-5', 'value-6', 'value-7', 'value-9', 'value-10', 'value-11', 'value-12']
    }
    NAME_OUTPUT_Y = 'label+1'
    LIST_DATA_HANDLING = ["normal", "smote"]

    MHA_LB = [-1]  # Lower bound for metaheuristics
    MHA_UB = [1]  # Upper bound for metaheuristics
    TEST_SIZE = 0.2         # Testing size
    VALID_SIZE = 0.2
    VALIDATION_USED = VALIDATION_USED

    METRICS_CLS_FOR_TESTING = ["AS", "PS", "RS", "F1S", "F2S", "NPV", "SS", "ROC", "GMS"]
    METRICS_REG_FOR_TESTING = ["RMSE", "MAE", "NSE", "KGE", "R2", "MAPE"]
    ## Training objective function
    OBJ_FUNCS = [  # Metric for training phase in network
        ## Binary-class
        # "binary_crossentropy",
        # "hinge",
        # "squared_hinge",

        ## Multiclass
        "categorical_crossentropy",
        # "categorical_hinge",
        # "kl_divergence",
    ]

    MI_RF_GRID = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                       'criterion': ["squared_error", "absolute_error"]}
    MI_SVM_GRID = {'C': [0.1, 1., 5., 10., 15.],
                  'gamma': ['scale', 'auto'],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    MI_LR_GRID = {'fit_intercept': [True, False]}
    MI_KNN_GRID = {'n_neighbors': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}
    MI_DT_GRID = {'criterion': ["squared_error", "absolute_error"]}
    MI_AdaBoost_GRID = {'n_estimators': [75, 100, 125, 150, 175, 200, 225, 250, 275, 300],
                       'loss': ['linear', 'square', 'exponential']}
    MI_MLP_GRID = {'hidden_layer_sizes': list(range(5, 21, 1)),
                       'activation': ['logistic', 'tanh', 'relu'],
                       'solver': ['lbfgs', 'sgd', 'adam'],
                       'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                       'max_iter': list(range(700, 1500, 100))}


    SVM_RF_GRID = {'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                       'criterion': ["squared_error", "absolute_error"]}
    SVM_SVM_GRID = {'C': [0.1, 1., 5., 10., 15.],
                  'gamma': ['scale', 'auto'],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    SVM_LR_GRID = {'fit_intercept': [True, False]}
    SVM_KNN_GRID = {'n_neighbors': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]}
    SVM_DT_GRID = {'criterion': ["squared_error", "absolute_error"]}
    SVM_AdaBoost_GRID = {'n_estimators': [50, 100, 150, 200, 250, 300, 350, 400, 450, 500],
                       'loss': ['linear', 'square', 'exponential']}
    SVM_MLP_GRID = {'hidden_layer_sizes': list(range(7, 26, 1)),
                       'activation': ['logistic', 'tanh', 'relu'],
                       'solver': ['lbfgs', 'sgd', 'adam'],
                       'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                       'max_iter': list(range(700, 1500, 100))}


    RF_RF_GRID = {'n_estimators': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50],
                       'criterion': ["squared_error", "absolute_error"]}
    RF_SVM_GRID = {'C': [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5],
                  'gamma': ['scale', 'auto'],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    RF_LR_GRID = {'fit_intercept': [True, False]}
    RF_KNN_GRID = {'n_neighbors': list(range(3, 26))}
    RF_DT_GRID = {'criterion': ["squared_error", "absolute_error"]}
    RF_AdaBoost_GRID = {'n_estimators': list(range(30, 210, 10)),
                       'loss': ['linear', 'square', 'exponential']}
    RF_MLP_GRID = {'hidden_layer_sizes': list(range(3, 16, 1)),
                       'activation': ['logistic', 'tanh', 'relu'],
                       'solver': ['lbfgs', 'sgd', 'adam'],
                       'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                       'max_iter': list(range(700, 1500, 100))}


    MCFS_RF_GRID = {'n_estimators': list(range(10, 210, 10)),
                       'criterion': ["squared_error", "absolute_error"]}
    MCFS_SVM_GRID = {'C': [0.1, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
                  'gamma': ['scale', 'auto'],
                  'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    MCFS_LR_GRID = {'fit_intercept': [True, False]}
    MCFS_KNN_GRID = {'n_neighbors': list(range(5, 26, 1))}
    MCFS_DT_GRID = {'criterion': ["squared_error", "absolute_error"]}
    MCFS_AdaBoost_GRID = {'n_estimators': list(range(30, 210, 10)),
                       'loss': ['linear', 'square', 'exponential']}
    MCFS_MLP_GRID = {'hidden_layer_sizes': list(range(4, 26, 1)),
                       'activation': ['logistic', 'tanh', 'relu'],
                       'solver': ['lbfgs', 'sgd', 'adam'],
                       'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                       'max_iter': list(range(700, 1500, 100))}
