import pandas as pd
import numpy as np
import keras
from keras.utils import Sequence
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
import argparse
import os
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import binary_crossentropy
from keras.optimizers import Adadelta
from keras.layers import GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet

from config import BATCH_SIZE, SHAPE, SHAFFLE_FLAG, USE_CACHE_FLAG
from dataset_processing import get_splited_dataset
from keras_model import ProteinDataGenerator, ProteinModel


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train_dataset_table',
                            help='Path to csv file with train dataset.')
    arg_parser.add_argument('--train_dataset_folder',
                            help='Path to folder with train images.')
    arg_parser.add_argument('--epochs', default=5, type=int,
                            help='Number of epochs.')
    arg_parser.add_argument('--checkpoints', help='Path to save model weights.')
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    train_dataset, val_dataset = get_splited_dataset(args.train_dataset_folder,
                                                     args.train_dataset_table)

    train_generator = ProteinDataGenerator(train_dataset[0], train_dataset[1],
                                           BATCH_SIZE, SHAPE, SHAFFLE_FLAG,
                                           USE_CACHE_FLAG)

    val_generator = ProteinDataGenerator(val_dataset[0], val_dataset[1],
                                         BATCH_SIZE, SHAPE, SHAFFLE_FLAG,
                                         USE_CACHE_FLAG)

    model = ProteinModel(shape=SHAPE)
    model.build_model()
    model.compile_model()
    model.set_generators(train_generator, val_generator)
    model.get_summary()

    model.learn(args.epochs, args.checkpoints)

