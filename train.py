# using keras and tensorflow
import os
import argparse
import cv2
import numpy as np

import tensorflow
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument("train_inp", type=str, help="folder path of training frame sequence")
parser.add_argument("train_target", type=str, help="file path of training speed sequence")

args = parser.parse_args()
train_inp    = os.path.abspath(args.train_inp)
train_target = os.path.abspath(args.train_target)

# functions...
def get_batch(train_inp, train_target, start, length=128, format="frame%6d.png"):
    inp = []
    for count in range(length):
        print("\rimage {} of {}".format(count + 1, length), end="")
        diff = cv2.imread(os.path.join(train_inp, (format % count).replace(" ", "0")))
        mean, stdev = cv2.meanStdDev(diff)
        mean = mean[0][0]
        stdev = stdev[0][0]
        diff = diff - mean
        diff = diff * (np.float32(1 / stdev))
        inp.append(diff)
    print()

    with open(train_target, "r") as speeds:
        target = list(map(float, speeds.readlines()[start: start+length]))

    inp = np.array(inp)
    target = np.array(target)

    return inp, target

def get_data(train_inp, train_target, format="frame%6d.png"):
    inp = []

    print("reading speeds...")
    with open(train_target, "r") as speeds:
        target = speeds.readlines()

    print("reading images...")
    for count in range(len(target)):
        print("\r {}".format(count), end="")
        diff = cv2.imread(os.path.join(train_inp, (format % count).replace(" ", "0")))
        mean, stdev = cv2.meanStdDev(diff)
        diff = diff - mean
        diff = diff * (np.float32(1 / stdev))
        inp.append(diff)

    print()

    inp = np.array(inp)
    target = np.array(target)

    return inp, target

def build_model(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(5, 5)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(5, 5)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(5, 5)))
    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation=None))

    model.compile(loss="mse", optimizer="Adam")

    print(model.summary())

    return model

input_shape = get_batch(train_inp, train_target, 0, length=1)[0].shape[1:4]
print(input_shape)

try:
    model = keras.models.load_model("./weights.h5py")
    print("model loaded successfully!")
except:
    print("no model found... building model...")
    model = build_model(input_shape)

count = 0
while True:
    inp, target = get_batch(train_inp, train_target, count * 128, length=128)
    model.fit(inp, target, batch_size=16, epochs=1)
    print("saving weights...")
    model.save("./weights.h5py")
    count += 1

score = model.evaluate(x_test, y_test, batch_size=4)

# todo: implement dataset as tensorflow dataset object
