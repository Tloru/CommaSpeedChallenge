from . import *

import numpy as np

import tensorflow as tf # tensorflow >= 2.0
from tensorflow import keras

from keras.preprocessing import image
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model

# Extractor class courtesy @harvitronix:
# https://github.com/harvitronix/five-video-classification-methods/blob/master/extractor.py
# simplified to fit my needs
class Extractor:
    def __init__(self):
        base_model = InceptionV3(
            weights='imagenet',
            include_top=True
        )

        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer('avg_pool').output
        )

    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        inp = image.img_to_array(img)
        inp = np.expand_dims(inp, axis=0)
        inp = preprocess_input(inp)

        features = self.model.predict(inp)
        return features[0]

class RNNCore:
    def __init__(self, inp_size, out_size, timesteps=32, batch_size=128, weights=None):
        self.inp_size = inp_size
        self.out_size = out_size
        self.timesteps = timesteps
        self.batch_size = batch_size

        if weights is None: self.model = self._build_model()
        else: self.model = load_model(weights)

    def _build_model():
        model = keras.models.Sequential()

        # input layer
        model.add(keras.layers.LSTM(
            256, return_sequences=True, stateful=True,
            batch_input_shape=(self.batch_size, self.timesteps, self.input_size)))

        # hidden RNN layers
        model.add(keras.layers.LSTM(256, return_sequences=True, stateful=True))
        model.add(keras.layers.LSTM(256, return_sequences=True, stateful=True))

        # stateful → non-stateful
        model.add(keras.layers.LSTM(256, stateful=True))

        # dense hidden layers
        model.add(keras.layers.Dense(32, activation="relu"))
        model.add(keras.layers.Dense(32, activation="relu"))

        # output
        model.add(keras.layers.Dense(self.out_size, activation=None))

        model.compile(loss='mse', optimizer='rmsprop')

        return model

# TODO: implement infinite image dataset
class InfiniSet:
    pass

def loader(count, total, length=32):
    bar = '█' * int(round((length * count) / total)) + ' ' * (bar_length - filled_length)
    print("\r{}/{}: |{}| {}".format(count, total, bar, round((count / total) * 100, ndigits=1)))
