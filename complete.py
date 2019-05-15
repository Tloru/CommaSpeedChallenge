import numpy as np
import os
import statistics as stat

import cv2

import tensorflow as tf # tensorflow >= 2.0
from tensorflow import keras
from keras.preprocessing import image
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model, Sequential
from keras.layers import RNN, Dense, LSTM

# Extractor class courtesy @harvitronix:
# https://github.com/harvitronix/five-video-classification-methods/blob/master/extractor.py
# simplified to fit my needs
class Extractor:
    def __init__(self):
        base_model = InceptionV3(
            weights='imagenet',
            include_top=True
        )

        print(base_model.get_layer("avg_pool"))

        self.model = Model(
            inputs=base_model.input,
            outputs=base_model.get_layer("avg_pool").output
        )

        # self.model.compile(optimizer="Adam", loss="mse")

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

    def _build_model(self):
        model = Sequential()
        model.add(LSTM(32, return_sequences=True, stateful=True,
                       batch_input_shape=(self.batch_size, self.timesteps, self.inp_size)))
        model.add(LSTM(32, return_sequences=True, stateful=True))
        model.add(LSTM(32, stateful=True))
        model.add(Dense(1, activation=None))

        model.compile(loss='mse', optimizer='rmsprop')

        return model

def loader(count, total, length=32):
    filled = int(round((length * count) / total))
    bar = '█' * filled + ' ' * (length - filled)
    print("\r{}/{}: |{}| {}%".format(count, total, bar, round((count / total) * 100, ndigits=1)), end="")

def feature_generator(extractor, inp_video, frame_folder):
    frames = cv2.VideoCapture(inp_video)
    success, frame = frames.read()

    while success:
        # process frame
        frame = frame[:][20:320]
        # TODO: resize to (299, 299)

        # useless read/write... working on workaround
        cv2.imwrite(os.path.join(frame_folder, "frame.png"), frame)
        extraction = extractor.extract(os.path.join(frame_folder, "frame.png"))

        yield extraction

def read_data(out_text):
    with open(out_text, "r") as text:
        out = text.readlines()

    out = list(map(float, out))

    mean = stat.mean(out)
    stdev = stat.stdev(out)

    out = list(map(lambda x: (x - mean) / stdev, out))
    print("output data has been normalized with a mean of {} and a stdev of {}".format(mean, stdev))

    return out, mean, stdev

def train(inp_video, frame_folder, out_text, use_cached=False):
    # set up models
    extractor = Extractor()
    inp_size = extractor.model.layers[-1].output_shape[1]
    out_size = 1
    batch_size = 128
    rnn_core = RNNCore(inp_size, out_size, batch_size=batch_size)

    print(extractor.extract("./data/shot.png").shape)

    inp = feature_generator(extractor, inp_video, frame_folder)
    out, mean, stdev = read_data(out_text)

    data_gen = TimeseriesGenerator(inp, out,
                               length=32, sampling_rate=2,
                               batch_size=2)

    rnn_core.model.fit(train, steps_per_epoch=2048, validation_split=0.2, shuffle=False, batch_size=batch_size, epochs=10)

if __name__ == "__main__":
    train("./data/train.mp4", "./train/", "./data/train.txt", use_cached=True)
