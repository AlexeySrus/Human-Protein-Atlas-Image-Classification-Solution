import pandas as pd
import numpy as np
import keras
from keras.utils import Sequence
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook
import os
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import binary_crossentropy
from keras.optimizers import Adadelta
from keras.layers import GlobalAveragePooling2D
from keras.applications.mobilenet import MobileNet
from keras.applications.inception_v3 import InceptionV3
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
import keras.backend as K


def load_image(path):
    """
    Load image from path
    Args:
        path: image path

    Returns:
        image in (0, 1) range
    """
    R = Image.open(path + '_red.png')
    G = Image.open(path + '_green.png')
    B = Image.open(path + '_blue.png')
    Y = Image.open(path + '_yellow.png')

    im = np.stack((
        np.array(R),
        np.array(G),
        np.array(B),
        np.array(Y)), -1)

    im = np.divide(im, 255)
    return im


class ProteinDataGenerator(keras.utils.Sequence):
    """ Standart data generator for keras model """
    def __init__(self, paths, labels, batch_size, shape, shuffle=False,
                 use_cache=False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        if use_cache == True:
            self.cache = np.zeros(
                (paths.shape[0], shape[0], shape[1], shape[2]))
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[
                  idx * self.batch_size: (idx + 1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros(
            (paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(
                    paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]

        return X, y

    def on_epoch_end(self):

        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __load_image(self, path):
        return load_image(path)


def task_f1(y_true, y_pred):
    """
    Tesor metric function
    Args:
        y_true: true value
        y_pred: predicted value

    Returns:
        Mean F1-measure
    """
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def task_f1_loss(y_true, y_pred):
    """
    Tensor loss function
    Args:
        y_true: true value
        y_pred: predicted value

    Returns:
        Mean F1-measure
    """
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


class ProteinModel:
    """ Model class """
    def __init__(self, num_classes=28, shape=(512, 512, 3)):
        self.num_classes = num_classes
        self.img_rows = shape[0]
        self.img_cols = shape[1]
        self.input_shape = (self.img_rows, self.img_cols, shape[2])
        self.last_epohs=0

    def build_model(self, _bmodel=InceptionV3):
        """
        Method for model building
        Args:
            _bmodel: Base feature extractor model

        Returns:
        """
        input_img = Input(shape=self.input_shape)

        base_model = _bmodel(include_top=False,
                               input_shape=self.input_shape,
                               classes=self.num_classes,
                               weights=None)

        base_model.trainable = True

        inp = Input(shape=(base_model.output_shape[1],
                           base_model.output_shape[2],
                           base_model.output_shape[3]))
        x = GlobalAveragePooling2D()(inp)
        m = Dense(self.num_classes, activation='sigmoid')(x)

        top_model = Model(inp, m, 'BaseModel')

        self.model = Model(input_img,
                           top_model(base_model(input_img)),
                           'ProteinModel')

    def compile_model(self, optimizer='adam'):
        self.model.compile(loss=keras.losses.binary_crossentropy,
                           optimizer=optimizer,
                           metrics=['accuracy', task_f1])

    def set_generators(self, train_generator, validation_generator):
        self.training_generator = train_generator
        self.validation_generator = validation_generator

    def learn(self, epochs=1, checkpoint_dir=None):
        callbacks = []

        if checkpoint_dir is not None:
            if not os.path.isdir(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            callbacks.append(ModelCheckpoint(
                os.path.join(checkpoint_dir, 'all-{epoch}.h5'.format(
                    epoch='{epoch}')),
                save_weights_only=True))

        return self.model.fit_generator(generator=self.training_generator,
                                        validation_data=self.validation_generator,
                                        use_multiprocessing=True,
                                        workers=8,
                                        callbacks=callbacks,
                                        epochs=epochs,
                                        initial_epoch=self.last_epohs)

    def score(self):
        return self.model.evaluate_generator(
            generator=self.validation_generator,
            use_multiprocessing=True,
            workers=8)

    def predict(self, data_generator):
        return self.model.predict_generator(generator=data_generator,
                                            use_multiprocessing=True)

    def get_summary(self):
        return self.model.summary()

    def load_weights(self, weights_tuple=None):
        if weights_tuple is not None:
            self.model.load_weights(weights_tuple[0])
            self.last_epohs = weights_tuple[1]
