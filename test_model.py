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
import numpy as np

from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from config import (BATCH_SIZE, SHAPE,
                    SHAFFLE_FLAG, USE_CACHE_FLAG, TRESHOLDS_PATH, BASE_MODEL)
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


def get_labels_by_prediction(predicion, tresholds):
    results_labels = []

    assert len(predicion) == len(tresholds)

    for i in range(len(predicion)):
        if predicion[i] >= tresholds[i]:
            results_labels.append(i)

    return np.array(results_labels)


if __name__ == '__main__':
    args = parse_arguments()

    model = ProteinModel(shape=SHAPE)
    model.build_model(BASE_MODEL)
    model.compile_model()
    model.load_weights((args.model_path, 0))
    model.get_summary()

    submit = pd.read_csv(args.test_table)

    try:
        thresholds = np.load(TRESHOLDS_PATH)
    except:
        thresholds = np.array([0.2]*28)

    print('Tresholds: {}'.format(thresholds))

    predicted = []
    for name in tqdm(submit['Id']):
        path = os.path.join(args.test_folder, name)
        image = load_image(path)
        score_predict = model.model.predict(image[np.newaxis])[0]
        label_predict = get_labels_by_prediction(score_predict, thresholds)
        #label_predict = np.arange(28)[score_predict >= 0.2]
        str_predict_label = ' '.join(str(l) for l in label_predict)
        predicted.append(str_predict_label)


    submit['Predicted'] = predicted
    submit.to_csv(args.results_path, index=False)
