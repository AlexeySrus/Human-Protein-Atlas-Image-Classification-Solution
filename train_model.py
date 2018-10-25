import pandas as pd
import numpy as np
import keras
from keras.utils import Sequence
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
import argparse
import os
import re
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import binary_crossentropy
from keras.optimizers import Adadelta
from keras.layers import GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from config import BATCH_SIZE, SHAPE, SHAFFLE_FLAG, USE_CACHE_FLAG
from dataset_processing import get_splited_dataset
from keras_model import ProteinDataGenerator, ProteinModel


def get_last_epoch_weights_path(checkpoints_dir):
    """
    Get last epochs weights from target folder
    Args:
        checkpoints_dir: target folder

    Returns:
        (Path to current weights file, current epoch number)
    """
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)
        return None

    weights_files_list = [
        matching_f.group()
        for matching_f in map(
            lambda x: re.match('all-\d+.h5', x),
            os.listdir(checkpoints_dir)
        ) if matching_f if not None
    ]

    if len(weights_files_list) == 0:
        return None

    weights_files_list.sort(key=lambda x: -int(x.split('-')[1].split('.')[0]))


    print('LOAD MODEL PATH: {}'.format(
        os.path.join(checkpoints_dir, weights_files_list[0])
    ))

    return os.path.join(checkpoints_dir,
                        weights_files_list[0]
                        ), int(
        weights_files_list[0].split('-')[1].split('.')[0]
    )


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train-table',
                            help='Path to csv file with train dataset.')
    arg_parser.add_argument('--train-folder',
                            help='Path to folder with train images.')
    arg_parser.add_argument('--epochs', default=5, type=int,
                            help='Number of epochs.')
    arg_parser.add_argument('--checkpoints', help='Path to save model weights.')
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    train_dataset, val_dataset = get_splited_dataset(args.train_folder,
                                                     args.train_table)

    train_generator = ProteinDataGenerator(train_dataset[0], train_dataset[1],
                                           BATCH_SIZE, SHAPE, SHAFFLE_FLAG,
                                           USE_CACHE_FLAG)

    val_generator = ProteinDataGenerator(val_dataset[0], val_dataset[1],
                                         BATCH_SIZE, SHAPE, SHAFFLE_FLAG,
                                         USE_CACHE_FLAG)

    model = ProteinModel(shape=SHAPE)
    model.build_model(InceptionV3)
    model.compile_model()
    model.load_weights(get_last_epoch_weights_path(args.checkpoints))
    model.set_generators(train_generator, val_generator)
    model.get_summary()

    model.learn(args.epochs, args.checkpoints)

