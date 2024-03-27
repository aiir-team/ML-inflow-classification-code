# !/usr/bin/env python
# Created by "Thieu" at 18:20, 08/12/2021 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
import numpy as np
from config import Config, Const
from utils.io_util import load_dataset
from imblearn.over_sampling import SMOTE
import pandas as pd


class MiniBatch:
    def __init__(self, X_train, y_train, batch_size):
        self.X_train = X_train
        self.y_train = y_train
        self.batch_size = batch_size

    def random_mini_batches(self, seed_number=None):
        X, Y = self.X_train.T, self.y_train.T
        mini_batch_size = self.batch_size

        m = X.shape[1]  # number of training examples
        mini_batches = []
        np.seed(seed_number)

        # Step 1: Shuffle (X, Y)
        permu = list(np.permutation(m))
        shuffled_X = X[:, permu]
        shuffled_Y = Y[:, permu].reshape((Y.shape[0], m))

        # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
        num_complete_minibatches = int(np.floor(m / mini_batch_size))  # number of mini batches of size mini_batch_size in your partitionning
        for k in range(0, num_complete_minibatches):
            mini_batch_X = shuffled_X[:, k * mini_batch_size: (k + 1) * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, k * mini_batch_size: (k + 1) * mini_batch_size]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        # Handling the end case (last mini-batch < mini_batch_size)
        if m % mini_batch_size != 0:
            mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
            mini_batch = (mini_batch_X, mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches


def get_scaler(mode:str, X_data:None, lb=None, ub=None):
    """
    mode = "dataset" --> Get scaler based on input X
    mode = "lbub" --> get scaler based on lower bound, upper bound in phase 2
    """
    scaler = MinMaxScaler()  # Data scaling using the MinMax method
    if mode == "lbub":
        if lb is None or ub is None:
            print("Lower bound and upper bound for lbub scaling method are required!")
            exit(0)
        lb = np.squeeze(np.array(lb))
        ub = np.squeeze(np.array(ub))
        X_data = np.array([lb, ub])
        scaler.fit(X_data)
    else:           # mode == "dataset":
        scaler.fit(X_data)
    return scaler



def transfer_hinge_dict(data, scaler=None):
    data["scale_Y"][np.where(data["scale_Y"] == 0)] = -1
    data["Y_train"][np.where(data["Y_train"] == 0)] = -1
    data["Y_test"][np.where(data["Y_test"] == 0)] = -1
    return data


def transform_hinge_list(data, scaler=None):
    data = np.squeeze(np.array(data))
    data[np.where(data == 0)] = -1
    return data


def invert_hinge_list(data, scaler=None):
    data = np.squeeze(np.array(data))
    data = np.ceil(data).astype(int)
    data[np.where(data == -1)] = 0
    return data


def transform_sigmoid_list(data, scaler=None):
    return data


def invert_sigmoid_list(data, scaler=None):
    data = np.squeeze(np.array(data))
    data = np.rint(data).astype(int)
    return data


def transform_softmax_list(data, scaler=None):
    data = scaler.transform(np.reshape(data, (-1, 1)))
    return data


def invert_softmax_list(data, scaler=None):
    data = np.squeeze(np.array(data))
    return np.argmax(data, axis=1)


def transform_one_hot_to_label(data):
    data = np.squeeze(np.array(data))
    return np.argmax(data, axis=1)


def get_label_scaler_and_inverter(obj="sigmoid", y_data=None):
    """
    Return: label scaler function, invert label scaler function, one-hot-encoder object
    """
    if obj in Const.TANH_LOSSES:
        return transform_hinge_list, invert_hinge_list, None
    elif obj in Const.SIGMOID_LOSSES:
        return transform_sigmoid_list, invert_sigmoid_list, None
    else:
        ohe_scaler = OneHotEncoder(sparse=False)
        ohe_scaler.fit(np.reshape(y_data, (-1, 1)))
        return transform_softmax_list, invert_softmax_list, ohe_scaler


def get_label_scaler_and_inverter_elm(obj="sigmoid", y_data=None):
    """
        This is for ELM only
        Return: label scaler function, invert label scaler function, one-hot-encoder object
    """
    if obj in Const.TANH_LOSSES or obj in Const.SIGMOID_LOSSES:
        return transform_sigmoid_list, invert_sigmoid_list, None
    else:
        ohe_scaler = OneHotEncoder(sparse=False)
        ohe_scaler.fit(np.reshape(y_data, (-1, 1)))
        return transform_softmax_list, invert_softmax_list, ohe_scaler


def transform_list_data(data_list, func, paras):
    data_new = [None if data is None else func(data, paras) for data in data_list]
    return data_new


def get_centers_data(data, n_nodes=3, method="kmean"):
    if method == "kmean":
        kobj = KMeans(n_clusters=n_nodes, init='random', random_state=11).fit(data)
        return kobj.cluster_centers_
    elif method == "random":
        random_args = np.random.choice(len(data), n_nodes, replace=False)
        return data[random_args]


def split_dataset(dataset, input_x, output_y='label+1', scaler="std"):
    X = dataset[input_x].values
    Y = dataset[output_y].values
    lb_encoder = LabelEncoder()
    y = lb_encoder.fit_transform(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=Config.VALID_SIZE, random_state=Config.SEED)
    if scaler == "std":
        scaler = StandardScaler()
        scaler.fit(x_train)
    else:
        scaler = MinMaxScaler()
        scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, scaler, lb_encoder


def split_smote_dataset(dataset, input_x, output_y='label+1', scaler="std"):
    X = dataset[input_x].values
    Y = dataset[output_y].values
    lb_encoder = LabelEncoder()
    y = lb_encoder.fit_transform(Y)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=Config.VALID_SIZE, random_state=Config.SEED)
    if scaler == "std":
        scaler = StandardScaler()
        scaler.fit(x_train)
    else:
        scaler = MinMaxScaler()
        scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    smote = SMOTE(random_state=Config.SEED)
    x_smote, y_smote = smote.fit_resample(x_train, y_train)
    return x_smote, x_test, y_smote, y_test, scaler, lb_encoder


def split_dataset_regression(X, y, scaler="std"):
    idx = int(len(X) * (1 - Config.VALID_SIZE))
    x_train, x_test, y_train, y_test = X[:idx], X[idx:], y[:idx], y[idx:]
    # x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=Config.VALID_SIZE, random_state=Config.SEED, shuffle=False)
    if scaler == "std":
        scaler_X = StandardScaler()
        scaler_X.fit(x_train)
        scaler_y = StandardScaler()
        scaler_y.fit(y_train.reshape(-1, 1))
    else:
        scaler_X = MinMaxScaler()
        scaler_X.fit(x_train)
        scaler_y = MinMaxScaler()
        scaler_y.fit(y_train.reshape(-1, 1))
    x_train = scaler_X.transform(x_train)
    x_test = scaler_X.transform(x_test)
    y_train = scaler_y.transform(y_train.reshape(-1, 1))
    y_test = scaler_y.transform(y_test.reshape(-1, 1))
    return x_train, x_test, y_train, y_test, scaler_X, scaler_y, idx


def convert_to_classification(y=None, month=None, matrix="mean"):
    month = np.asarray(month, dtype=int)
    if matrix == "mean":
        matrix_value = np.array([
            [1.90, 2.50, 3.15, 4.15],
            [0.80, 1.35, 1.95, 3.00],
            [0.55, 1.05, 1.70, 2.80],
            [0.30, 0.65, 1.15, 2.10],
            [0.65, 0.95, 1.35, 2.05],
            [0.90, 1.25, 1.65, 2.35],
            [2.80, 3.70, 4.75, 6.55],
            [15.05, 17.60, 20.40, 24.75],
            [18.55, 21.20, 24.05, 28.35],
            [11.30, 13.35, 15.60, 19.05],
            [4.75, 5.95, 7.30, 9.50],
            [2.70, 3.45, 4.30, 5.65]
        ])
    else:
        matrix_value = np.array([
            [1.6, 2.5, 4.15, 4.8],
            [0.5, 1.35, 3.0, 3.7],
            [0.3, 1.05, 2.8, 3.5],
            [0.1, 0.65, 1.15, 2.1],
            [0.5, 0.95, 2.05, 2.5],
            [0.7, 1.25, 2.35, 2.8],
            [2.3, 3.7, 6.55, 7.7],
            [13.7, 17.6, 24.75, 27.5],
            [17.1, 21.2, 28.35, 31.0],
            [10.2, 13.35, 19.05, 21.2],
            [4.1, 5.95, 9.5, 10.9],
            [2.3, 3.45, 5.65, 6.5]
        ])
    y1 = []
    for idx in range(len(y)):
        thesh = matrix_value[month[idx] - 1, :]
        x = y[idx]
        if x < thesh[0]:
            temp = "Low"
        elif thesh[0] <= x < thesh[1]:
            temp = "Medium Low"
        elif thesh[1] <= x < thesh[2]:
            temp = "Medium"
        elif thesh[2] <= x < thesh[3]:
            temp = "Medium High"
        else:
            temp = "High"
        y1.append(temp)
    return np.array(y1)
