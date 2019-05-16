import numpy as np
import os
import statistics as stat
import time

import cv2

import tensorflow as tf
from tensorflow import keras
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, Sequential
from keras.layers import Dense, GRU, Dropout

tf.logging.set_verbosity(tf.logging.FATAL)

class RNNCore:
    def __init__(self, inp_size, out_size, batch_size, step, weights=None):
        self.inp_size = inp_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.step = step

        if weights is None: self.model = self._build_model()
        else: self.model = load_model(weights)

    def _build_model(self):
        model = Sequential()

        model.add(GRU(
            256, return_sequences=True, stateful=True,
            batch_input_shape=(self.batch_size, self.step, self.inp_size))
        )
        model.add(GRU(128, return_sequences=True, stateful=True))
        model.add(GRU(64, return_sequences=True, stateful=True))
        model.add(GRU(32, stateful=True))
        model.add(Dropout(0.5))

        model.add(Dense(16, activation="relu"))
        model.add(Dense(self.out_size, activation='relu'))

        model.compile(loss='mean_squared_error', optimizer='rmsprop')
        model.summary()

        return model

class VideoSet:
    def __init__(self, video_path, text_path, extractor, batch_size, step):
        self.video = cv2.VideoCapture(video_path)

        with open(text_path, "r") as text:
            lines = list(map(float, text.readlines()))
            self.mean = stat.mean(lines)
            lines = list(map(lambda x: x / self.mean, lines))
            self.speeds = lines

        self.extractor = extractor
        self.batch_size = batch_size
        self.step = step
        self._count = 0
        self.data = self._data_shaper()

    def _video_encoder(self):
        while True:
            success, image = self.video.read()
            if not success: raise StopIteration("dataset is exahausted...")

            encoding = self.extractor.extract(image)
            speed = self.speeds[self._count]

            yield encoding, speed
            self._count += 1

    def _data_shaper(self):
        data = self._video_encoder()
        queue = []
        for step in range(self.step):
            inp, out = next(data)
            queue.append(inp)

        while True:
            yield np.array(queue), np.array([out])
            inp, out = next(data)
            queue = queue[1:] + [inp]

# Extractor class courtesy @harvitronix:
# https://github.com/harvitronix/five-video-classification-methods/blob/master/extractor.py
# modified to fit my needs
class Extractor:
    def __init__(self):
        base_model = InceptionV3(
            weights='imagenet',
            include_top=True
        )

        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer("avg_pool").output
        )

    def extract(self, image):
        inp = cv2.resize(image, (299, 299))
        inp = preprocess_input(inp)
        inp = np.expand_dims(inp, axis=0)

        features = self.model.predict(inp)
        return features[0]

def loader(count, total, length=32):
    filled = int(round((length * count) / total))
    bar = '█' * filled + ' ' * (length - filled)
    print("\r{}/{}: |{}| {}%".format(count, total, bar, round((count / total) * 100, ndigits=1)), end="")

if __name__ == "__main__":
    extractor = Extractor()

    inp_size = extractor.model.layers[-1].output_shape[1]
    out_size = 1
    batch_size = 120
    step = 32

    video_set = VideoSet(
        "./data/train.mp4",
        "./data/train.txt",
        extractor,
        batch_size,
        step
    )

    rnn_core = RNNCore(inp_size, out_size, batch_size, step)
    # try:
    #     rnn_core.model.load_weights("weights_one_epoch.h5py")
    #     print("Weights successfully loaded!")
    # except:
    #     print("Unable to load weights...")
    #     if input("Would you like to proceed? (Old weights might be overwritten.) (y/n)") != "y":
    #         raise RuntimeError("Weights could not be loaded and user chose to exit.")
    #     else:
    #         print("Starting with random weights (^C to exit)...")

    x_val, y_val = [], []

    print("Encoding validation data...")
    for count in range(batch_size * 5):
        x, y = next(video_set.data)

        x_val.append(x)
        y_val.append(y)

        loader(count + 1, batch_size * 5)
    print()

    while True:
        x_train, y_train = [], []

        print("Encoding training data...")
        for count in range(batch_size * 5):
            x, y = next(video_set.data)

            x_train.append(x)
            y_train.append(y)

            loader(count + 1, batch_size * 5)
        print()

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        rnn_core.model.fit(
            x_train, y_train,
            batch_size=batch_size, epochs=1, shuffle=False,
            validation_data=(x_val, y_val)
        )

        rnn_core.model.save_weights("./weights.h5py")

        print(video_set._count)
