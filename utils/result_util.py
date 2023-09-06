#!/usr/bin/env python
# Created by "Thieu" at 07:58, 02/11/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

from config import Const
from utils.io_util import save_to_csv_dict, save_results_to_csv
from utils.metric_util import my_classify_metrics
from utils.model_util import save_system
from utils.visual.curve import draw_loss_accuracy_curve, draw_roc_auc_multiclass, draw_precision_recall_multiclass, draw_roc_auc_curve, \
    draw_precision_recall_curve
from utils.visual.heatmap import draw_confusion_matrix


def save_results(results: dict, validation:True, metrics_head:dict, metrics_file: str,
                 test_filename: str, pathsave: str, name_labels=None,
                 name_model=None, n_labels=2, loss_train=None, system=None, verbose=False):
    ## Calculate performance metrics and save it to csv file
    matrix_conf_train, mm1 = my_classify_metrics(results[Const.Y_TRAIN_TRUE_UNSCALED], results[Const.Y_TRAIN_PRED_UNSCALED],
                              results[Const.Y_TRAIN_PRED_PROB], prefix="train", decimal=4)
    matrix_conf_test, mm2 = my_classify_metrics(results[Const.Y_TEST_TRUE_UNSCALED], results[Const.Y_TEST_PRED_UNSCALED],
                              results[Const.Y_TEST_PRED_PROB], prefix="test", decimal=4)
    if validation:
        matrix_conf_valid, mm3 = my_classify_metrics(results[Const.Y_VALID_TRUE_UNSCALED], results[Const.Y_VALID_PRED_UNSCALED],
                                  results[Const.Y_VALID_PRED_PROB], prefix="valid", decimal=4)
        mm = {**metrics_head, **mm1, **mm2, **mm3}
    else:
        mm = {**metrics_head, **mm1, **mm2}
    save_results_to_csv(mm, metrics_file, pathsave)

    ## Save prediction results of training set and testing set to csv file
    data = {key: results[key] for key in Const.HEADER_TRUTH_PREDICTED_TRAIN_FILE}
    save_to_csv_dict(data, f"{test_filename}-{Const.FILENAME_PRED_TRAIN}", pathsave)

    if validation:
        data = {key: results[key] for key in Const.HEADER_TRUTH_PREDICTED_VALID_FILE}
        save_to_csv_dict(data, f"{test_filename}-{Const.FILENAME_PRED_VALID}", pathsave)

    data = {key: results[key] for key in Const.HEADER_TRUTH_PREDICTED_TEST_FILE}
    save_to_csv_dict(data, f"{test_filename}-{Const.FILENAME_PRED_TEST}", pathsave)

    ## Visualization
    list_legends = ["Train", "Test"]
    draw_confusion_matrix(matrix_conf_train[0], name_labels, Const.TITLE_IMG_CONF_MATRIX_TRAIN,
                          f"{test_filename}-{Const.FILENAME_IMG_CM_TRAIN}", pathsave, Const.FILE_FIGURE_TYPES, verbose)
    if validation:
        list_legends = ["Train", "Validation"]
        draw_confusion_matrix(matrix_conf_valid[0], name_labels, Const.TITLE_IMG_CONF_MATRIX_VALID,
                              f"{test_filename}-{Const.FILENAME_IMG_CM_VALID}", pathsave, Const.FILE_FIGURE_TYPES, verbose)

    draw_confusion_matrix(matrix_conf_test[0], name_labels, Const.TITLE_IMG_CONF_MATRIX_TEST,
                          f"{test_filename}-{Const.FILENAME_IMG_CM_TEST}", pathsave, Const.FILE_FIGURE_TYPES, verbose)

    ## Save obj train to csv file
    if loss_train is not None:
        epoch = list(range(1, len(loss_train['loss']) + 1))
        data = {"epoch": epoch, "loss": loss_train['loss'], "val_loss": loss_train['val_loss'],
                "accuracy": loss_train['accuracy'], "val_accuracy": loss_train['val_accuracy']}
        save_to_csv_dict(data, f"{test_filename}-{Const.FILENAME_LOSS_TRAIN}", pathsave)
        draw_loss_accuracy_curve(loss_train, list_legends, None, f"{test_filename}-{Const.FILENAME_IMG_LOSS_ACC}",
                                 pathsave, Const.FILE_FIGURE_TYPES, verbose)

    ## Draw ROC-AUC curve and Precision/Recall curve
    if n_labels != 2:
        draw_roc_auc_multiclass(name_labels, results[Const.Y_TEST_TRUE_SCALED], results[Const.Y_TEST_PRED_PROB],
                                Const.TITLE_IMG_ROC_AUC_MULTI, f"{test_filename}-{Const.FILENAME_IMG_ROC_AUC}",
                                pathsave, Const.FILE_FIGURE_TYPES, verbose)
        draw_precision_recall_multiclass(name_labels, results[Const.Y_TEST_TRUE_SCALED], results[Const.Y_TEST_PRED_PROB],
                                         Const.TITLE_IMG_PRE_RECALL_MULTI, f"{test_filename}-{Const.FILENAME_IMG_PRE_RECALL}",
                                         pathsave, Const.FILE_FIGURE_TYPES, verbose)
    else:
        draw_roc_auc_curve(name_model, results[Const.Y_TEST_TRUE_SCALED], results[Const.Y_TEST_PRED_SCALED],
                           Const.TITLE_IMG_ROC_AUC, f"{test_filename}-{Const.FILENAME_IMG_ROC_AUC}",
                           pathsave, Const.FILE_FIGURE_TYPES, verbose)
        draw_precision_recall_curve(name_model, results[Const.Y_TEST_TRUE_SCALED], results[Const.Y_TEST_PRED_SCALED],
                                    Const.TITLE_IMG_PRE_RECALL_MULTI, f"{test_filename}-{Const.FILENAME_IMG_PRE_RECALL}",
                                    pathsave, Const.FILE_FIGURE_TYPES, verbose)
    ## Save models
    if system is not None:
        save_system(system, f"{pathsave}/{test_filename}-{Const.FILENAME_MODEL}", system.framework, system.hybrid_model)
