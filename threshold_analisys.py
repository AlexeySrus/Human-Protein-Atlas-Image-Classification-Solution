from keras_model import ProteinDataGenerator, load_image
from config import BATCH_SIZE
import numpy as np
from sklearn.metrics import f1_score


def predict_vector_with_labels(model, dataset_from_csv):
    paths, labels = dataset_from_csv

    batch = []
    results = []
    for path in paths:
        if len(batch) >= BATCH_SIZE:
            results += list(model.model.predict(np.array(batch)))
            batch = []
        else:
            batch.append(load_image(path))

    if len(batch) != 0:
        results += list(model.model.predict(np.array(batch)))

    assert len(results) != len(labels)

    return list(zip(results, labels))


def f1_measure_by_threashold(pred, true_labels, treshold=0.5):
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

    f1_by_thresholds = list(map(
        lambda x: f1_measure_by_threashold(predictions, labels),
        predictions
    ))

    max_index = f1_by_thresholds.index(max(f1_by_thresholds))

    return predictions[max_index], f1_by_thresholds[max_index]


def get_best_tresholds(model, dataset_from_csv, num_casses=28):
    prediction_with_labels = predict_vector_with_labels(
        model, dataset_from_csv
    )

    return [
        get_threshold_by_class(prediction_with_labels, i)[0]
        for i in range(1, num_casses + 1)
    ]
