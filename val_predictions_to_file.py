import argparse
from dataset_processing import get_splited_dataset
from config import BATCH_SIZE, SHAPE, SHAFFLE_FLAG, USE_CACHE_FLAG
from keras.applications.inception_v3 import InceptionV3
from keras_model import ProteinModel
from threshold_analisys import get_best_tresholds
import numpy as np

def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--train-table',
                            help='Path to csv file with train dataset.')
    arg_parser.add_argument('--train-folder',
                            help='Path to folder with train images.')
    arg_parser.add_argument('--model-path', help='Path to model weights.')
    arg_parser.add_argument('--result-path', help='Path to model prediction.')
    return arg_parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    val_dataset, _ = get_splited_dataset(args.train_folder, args.train_table)

    model = ProteinModel(shape=SHAPE)
    model.build_model(InceptionV3)
    model.compile_model()
    model.load_weights((args.model_path, 0))
    model.get_summary()

    best_thresholds = get_best_tresholds(model, val_dataset)

    np.save(args.result_path, np.array(best_thresholds))