from keras_model import ProteinDataGenerator, load_image
from config import BATCH_SIZE
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm


def labels_decoding(label_in_vector):
    labels = []
    for i in range(len(label_in_vector)):
        if label_in_vector[i] > 0.5:
            labels.append(i + 1)
    return labels


def predict_vector_with_labels(model, dataset_from_csv):
    paths, labels = dataset_from_csv

    results = []
    for path in tqdm(paths):
        results.append(list(model.model.predict(load_image(path)[np.newaxis]))[0])

    assert len(results) == len(labels)

    return list(zip(results, labels))


def f1_measure_by_threashold(pred, true_labels, treshold):
    pred_labels = list(map(lambda x: int(x > treshold), pred))
    return f1_score(pred_labels, true_labels)


def get_threshold_by_class(prediction_with_labels, num_class=1):
    pred_by_label = list(
        map(
            lambda x: (x[0][num_class - 1], int(num_class in x[1])),
            prediction_with_labels
        )
    )
    pred_by_label.sort(key=lambda x: -x[0])

    predictions = list(map(lambda x: x[0], pred_by_label))
    labels = list(map(lambda x: x[1], pred_by_label))

    tresholds = list(np.arange(0.01, 0.99, 0.01))

    f1_by_thresholds = list(map(
        lambda x: f1_measure_by_threashold(predictions, labels, x),
        tresholds
    ))

    max_index = f1_by_thresholds.index(max(f1_by_thresholds))

    return tresholds[max_index], f1_by_thresholds[max_index]


def get_best_tresholds(model, dataset_from_csv, num_casses=28):
    prediction_with_labels = predict_vector_with_labels(
        model, dataset_from_csv
    )

    pred_data = list(
        map(lambda x: (x[0], labels_decoding(x[1])), prediction_with_labels))

    return [
        get_threshold_by_class(pred_data, i)[0]
        for i in tqdm(range(1, num_casses + 1))
    ]
