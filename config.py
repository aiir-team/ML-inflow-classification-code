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
    FILENAME_PRED_TRAIN = "pred_train"
    FILENAME_PRED_VALID = "pred_valid"
    FILENAME_PRED_TEST = "pred_test"
    FILENAME_PRED_REAL_WORLD = "pred_real_world"

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
    DATA_RESULTS = f'{DATA_DIRECTORY}/results_anfis_cv'

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

    METRICS_FOR_TESTING_PHASE = ["AS", "PS", "RS", "F1S", "F2S", "MCC", "LS"]
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

class MhaConfig:
    EPOCH = [500]         # Number of generations or epoch in neural network and metaheuristics
    POP_SIZE = [50]     # Number of population size in metaheuristics

    elm = {
        "para": ["None"]
    }
    rbfn = {
        "epoch": EPOCH,
        "optimizer": ["adam",],    # https://keras.io/api/optimizers/         "RMSprop", "Adadelta", "SGD",
        "learning_rate": ["auto"],     # 0.001, 0.01, 0.1, or "auto"
        "batch_size": [16],
    }
    flnn = {
        "epoch": EPOCH,
        "optimizer": ["adam",],    # https://keras.io/api/optimizers/         "RMSprop", "Adadelta", "SGD",
        "learning_rate": ["auto"],     # 0.001, 0.01, 0.1, or "auto"
        "batch_size": [16],
    }
    anfis = {
        "epoch": EPOCH,
        "optimizer": ["Adam", "Adadelta", "Adagrad", "SGD", "Adamax", ],    # https://keras.io/api/optimizers/         "RMSprop", "Adadelta", "SGD", "Adam"
        "learning_rate": [0.0001, 0.00025, 0.0005, 0.001, 0.0025, 0.005,
                          0.01, 0.025, 0.05, 0.1, 0.15, 0.2],     # 0.001, 0.01, 0.1, or "auto"
        "batch_size": [8],
    }
