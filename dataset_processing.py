import pandas as pd
import numpy as np
import keras
from keras.utils import Sequence
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
import os


def split_dataset(dataset, val_split, test_split):
    """
    Dataset split function
    Args:
        dataset: original dataset
        val_split: validation rate from dataset (mast in range [0, 1])
        test_split: test rate from dataset (mast in range [0, 1])

    Returns:
        train, validation, test datasets
    """
    val_dataset = dataset.sample(frac=val_split, replace=False)

    k = 1 - len(val_dataset) / len(dataset)

    test_dataset = \
        dataset.loc[set(dataset.index) - \
                    set(val_dataset.index)].sample(frac=test_split / k,
                                                   replace=False)

    train_index = set(dataset.index) - set(val_dataset.index) - \
                  set(test_dataset.index)

    train_dataset = dataset.loc[train_index]

    return train_dataset, val_dataset, test_dataset


def get_dataset_by_csv(path_to_train_folder, train_csv):
    """
    Extract data from csv table
    Args:
        path_to_train_folder: path to folder with train data
        train_csv: path to train csv table

    Returns:
        (Full paths to data, data labels)
    """
    path_to_train = path_to_train_folder
    data = train_csv

    paths = []
    labels = []

    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


def get_splited_dataset(path_to_train_folder, path_to_train_csv,
                        val_split_rate=0.25):
    """
    Get splitted and extracted dataset
    Args:
        path_to_train_folder:
        path_to_train_csv:
        val_split_rate:

    Returns:
        (train dataset, validation dataset)
    """
    train_data = pd.read_csv(path_to_train_csv)
    train_dataset, val_dataset, test_dataset = split_dataset(train_data,
                                                             val_split_rate, 0)
    return get_dataset_by_csv(path_to_train_folder,
                              train_dataset), get_dataset_by_csv(
        path_to_train_folder, val_dataset)