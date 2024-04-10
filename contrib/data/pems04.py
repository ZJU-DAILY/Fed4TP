from federatedscope.register import register_data
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import DataLoader
from dateutil import parser
import pandas as pd
import numpy as np
import torch
import math


def load_my_data(config, num_nodes):
    TIME_STEP = 12
    dataset = np.load(config.data.file_path + config.data.data_file).astype('float32')
    dataset = dataset[..., :1]

    # split to train test val
    dataset_size = dataset.shape[0]
    dataset_end_train = math.ceil(dataset_size * config.data.splits[0])
    dataset_end_test = math.ceil(dataset_size * (config.data.splits[0] + config.data.splits[1]))
    dataset_train = dataset[:dataset_end_train, :]
    dataset_test = dataset[dataset_end_train:dataset_end_test, :]
    dataset_val = dataset[dataset_end_test:, :]
    dataset_size_train = dataset_train.shape[0]
    dataset_size_test = dataset_test.shape[0]
    dataset_size_val = dataset_val.shape[0]

    # filter last batch
    dataset_train = dataset_train[:-(dataset_size_train % np.lcm(config.dataloader.batch_size, TIME_STEP + 1))]
    dataset_test = dataset_test[:-(dataset_size_test % np.lcm(config.dataloader.batch_size, TIME_STEP + 1))]
    dataset_val = dataset_val[:-(dataset_size_val % np.lcm(config.dataloader.batch_size, TIME_STEP + 1))]
    dataset_size_train = dataset_train.shape[0]
    dataset_size_test = dataset_test.shape[0]
    dataset_size_val = dataset_val.shape[0]

    dataset_dict = dict()
    num_nodes_per_client = math.floor(num_nodes / config.federate.client_num)

    if config.fedtfp.tm_use:
        dataset_size_per_time_window = math.floor(
            dataset_size_train / config.fedtfp.time_window_num / (TIME_STEP + 1)) * (TIME_STEP + 1)

    # split to different clients
    for client_idx in range(config.federate.client_num):
        # split nodes
        dataset_train_client = \
            dataset_train[:, client_idx * num_nodes_per_client:(client_idx + 1) * num_nodes_per_client, :]
        dataset_test_client = \
            dataset_test[:, client_idx * num_nodes_per_client:(client_idx + 1) * num_nodes_per_client, :]
        dataset_val_client = \
            dataset_val[:, client_idx * num_nodes_per_client:(client_idx + 1) * num_nodes_per_client, :]

        # scalar
        max_train = np.max(dataset_train_client[..., 0])
        max_test = np.max(dataset_test_client[..., 0])
        max_val = np.max(dataset_val_client[..., 0])
        min_train = np.min(dataset_train_client[..., 0])
        min_test = np.min(dataset_test_client[..., 0])
        min_val = np.min(dataset_val_client[..., 0])
        dataset_train_client[..., 0] = (dataset_train_client[..., 0] - min_train) / (max_train - min_train)
        dataset_test_client[..., 0] = (dataset_test_client[..., 0] - min_test) / (max_test - min_test)
        dataset_val_client[..., 0] = (dataset_val_client[..., 0] - min_val) / (max_val - min_val)

        # dataloader
        if config.fedtfp.tm_use:
            dataset_dict[client_idx + 1] = [dict() for _ in range(config.fedtfp.time_window_num)]
            for time_window in range(config.fedtfp.time_window_num):
                time_window_padding = time_window * dataset_size_per_time_window
                dataset_dict[client_idx + 1][time_window]['train'] = DataLoader(
                    [(dataset_train_client[i:i + TIME_STEP, :], dataset_train_client[i + TIME_STEP + 1, :])
                     for i in
                     range(time_window_padding, time_window_padding + dataset_size_per_time_window - TIME_STEP - 1,
                           TIME_STEP + 1)], config.data.batch_size, shuffle=False)
                dataset_dict[client_idx + 1][time_window]['test'] = DataLoader(
                    [(dataset_test_client[i:i + TIME_STEP, :], dataset_test_client[i + TIME_STEP + 1, :])
                     for i in range(0, dataset_size_test - TIME_STEP - 1, TIME_STEP + 1)],
                    config.data.batch_size, shuffle=False)
                dataset_dict[client_idx + 1][time_window]['val'] = DataLoader(
                    [(dataset_val_client[i:i + TIME_STEP, :], dataset_val_client[i + TIME_STEP + 1, :])
                     for i in range(0, dataset_size_val - TIME_STEP - 1, TIME_STEP + 1)],
                    config.data.batch_size, shuffle=False)
                dataset_dict[client_idx + 1][time_window]['scalar'] = {'train': {'max': max_train, 'min': min_train},
                                                          'test': {'max': max_test, 'min': min_test},
                                                          'val': {'max': max_val, 'min': min_val}}
        else:
            dataset_dict[client_idx + 1] = dict()
            dataset_dict[client_idx + 1]['train'] = DataLoader(
                [(dataset_train_client[i:i + TIME_STEP, ...], dataset_train_client[i + TIME_STEP:i + TIME_STEP + 1, :, :1])
                 for i in range(0, dataset_size_train - TIME_STEP, TIME_STEP + 1)],
                config.dataloader.batch_size, shuffle=False)
            dataset_dict[client_idx + 1]['test'] = DataLoader(
                [(dataset_test_client[i:i + TIME_STEP, ...], dataset_test_client[i + TIME_STEP:i + TIME_STEP + 1, :, :1])
                 for i in range(0, dataset_size_test - TIME_STEP, TIME_STEP + 1)],
                config.dataloader.batch_size, shuffle=False)
            dataset_dict[client_idx + 1]['val'] = DataLoader(
                [(dataset_val_client[i:i + TIME_STEP, ...], dataset_val_client[i + TIME_STEP:i + TIME_STEP + 1, :, :1])
                 for i in range(0, dataset_size_val - TIME_STEP, TIME_STEP + 1)],
                config.dataloader.batch_size, shuffle=False)
            dataset_dict[client_idx + 1]['scalar'] = {'train': {'max': max_train, 'min': min_train},
                                                      'test': {'max': max_test, 'min': min_test},
                                                      'val': {'max': max_val, 'min': min_val}}

    return dataset_dict, config


def call_my_data(config, client_cfgs=None):
    if config.data.type.lower() == "pems04":
        num_nodes = 307
        return load_my_data(config, num_nodes)


register_data("PeMS04", call_my_data)
