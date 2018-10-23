import pandas as pd
import numpy as np
import keras
from keras.utils import Sequence
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse
import os
import re

from config import BATCH_SIZE, SHAPE, SHAFFLE_FLAG, USE_CACHE_FLAG
from dataset_processing import get_splited_dataset
from keras_model import ProteinDataGenerator, ProteinModel, load_image


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--test-table',
                            help='Path to csv file with train dataset.')
    arg_parser.add_argument('--test-folder',
                            help='Path to test dataset folder.')
    arg_parser.add_argument('--model-path', help='Path to model weights.')
    arg_parser.add_argument('--results-path', help='Path to model prediction.')
    arg_parser.add_argument('--batch-size', type=int, default=16,
                            help='Batch size')
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    model = ProteinModel(shape=SHAPE)
    model.build_model()
    model.compile_model()
    model.load_weights((args.model_path, 0))
    model.get_summary()

    submit = pd.read_csv(args.test_table)

    predicted = []
    batch = []
    for name in tqdm(submit['Id']):
        if len(batch) < args.batch_size:
            path = os.path.join(args.test_folder, name)
            batch.append(load_image(path))
        else:
            #score_predict = model.model.predict(image[np.newaxis])[0]
            score_predictions = model.model.predict(np.array(batch))
            for score_predict in score_predictions:
                label_predict = np.arange(28)[score_predict >= 0.5]
                str_predict_label = ' '.join(str(l) for l in label_predict)
                predicted.append(str_predict_label)
            batch = []

    if len(batch) != 0:
        score_predictions = model.model.predict(np.array(batch))
        for score_predict in score_predictions:
            label_predict = np.arange(28)[score_predict >= 0.5]
            str_predict_label = ' '.join(str(l) for l in label_predict)
            predicted.append(str_predict_label)
        batch = []

    submit['Predicted'] = predicted
    submit.to_csv(args.results_path, index=False)