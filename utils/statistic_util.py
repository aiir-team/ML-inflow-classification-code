#!/usr/bin/env python
# Created by "Thieu" at 18:22, 21/04/2022 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np
from pathlib import Path
from pandas import read_csv, DataFrame
from sklearn.model_selection import ParameterGrid
from utils.io_util import save_to_csv_dict


def save_fast_to_csv(list_results, list_paths, columns):
    for idx, results in enumerate(list_results):
        df = DataFrame(results, columns=columns)
        df.to_csv(list_paths[idx], index=False)
    return True


def get_all_performance_metrics(list_pathread, trials, models, suf_fileread, cols_header_read, cols_header_save, filesave, pathsave):
    matrix_results = []
    for path_paras, pathread in list_pathread.items():
        for trial in range(trials):
            for model in models:
                path_model = f"{pathread}/trial-{trial}/{model['name']}"
                keys = model["param_grid"].keys()
                for mha_paras in list(ParameterGrid(model["param_grid"])):
                    # Load metrics
                    filename = "".join([f"-{mha_paras[key]}" for key in keys])
                    filepath = f"{path_model}/{filename[1:]}-{suf_fileread}.csv"
                    df = read_csv(filepath, usecols=cols_header_read)
                    values = df.values.tolist()[0]
                    results = [path_paras, trial, model['name']] + values
                    matrix_results.append(np.array(results))
    matrix_results = np.array(matrix_results)
    matrix_dict = {}
    for idx, key in enumerate(cols_header_save):
        matrix_dict[key] = matrix_results[:, idx]
    ## Save final file to csv
    save_to_csv_dict(matrix_dict, filesave, pathsave)
    # savetxt(f"{Config.DATA_RESULTS}/statistics_final.csv", matrix_results, delimiter=",")
    df = read_csv(f"{pathsave}/{filesave}.csv", usecols=cols_header_save)
    return df


def calculate_statistics(df_results, path_paras, models, cols_header_read, pathsave, list_filenames, cols_header_save):
    min_results, mean_results, max_results, std_results, cv_results = [], [], [], [], []
    Path(pathsave).mkdir(parents=True, exist_ok=True)
    for model in models:
        keys = model["param_grid"].keys()
        for mha_paras in list(ParameterGrid(model["param_grid"])):
            model_paras = "".join([f"-{mha_paras[key]}" for key in keys])
            model_paras = model_paras[1:]
            df_result = df_results[(df_results["path_paras"] == path_paras) & (df_results["model"] == model["name"]) &
                                   (df_results["model_paras"] == model_paras)][cols_header_read]

            t1 = df_result.min(axis=0).to_numpy()
            t2 = df_result.mean(axis=0).to_numpy()
            t3 = df_result.max(axis=0).to_numpy()
            t4 = df_result.std(axis=0).to_numpy()
            t5 = t4 / t2

            t1 = [path_paras, model["name"], model_paras] + t1.tolist()
            t2 = [path_paras, model["name"], model_paras] + t2.tolist()
            t3 = [path_paras, model["name"], model_paras] + t3.tolist()
            t4 = [path_paras, model["name"], model_paras] + t4.tolist()
            t5 = [path_paras, model["name"], model_paras] + t5.tolist()

            min_results.append(t1)
            mean_results.append(t2)
            max_results.append(t3)
            std_results.append(t4)
            cv_results.append(t5)
    save_fast_to_csv([min_results, mean_results, max_results, std_results, cv_results],
                     [f"{pathsave}/{list_filenames[0]}", f"{pathsave}/{list_filenames[1]}",
                      f"{pathsave}/{list_filenames[2]}", f"{pathsave}/{list_filenames[3]}", f"{pathsave}/{list_filenames[4]}"],
                     columns=cols_header_save)
