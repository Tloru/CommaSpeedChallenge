# using keras and tensorflow
import os
import argparse
import cv2
import numpy as np

import tensorflow
from tensorflow import keras

parser = argparse.ArgumentParser()
parser.add_argument("test_inp", type=str, help="folder path of training frame sequence")

args = parser.parse_args()
test_inp    = os.path.abspath(args.test_inp)

# functions...
def get_batch(test_inp, start, length=128, format="frame%6d.png"):
    inp = []
    for count in range(length):
        print("\rimage {} of {}".format(count + 1, length), end="")
        diff = cv2.imread(os.path.join(test_inp, (format % count).replace(" ", "0")))
        mean, stdev = cv2.meanStdDev(diff)
        mean = mean[0][0]
        stdev = stdev[0][0]
        diff = diff - mean
        diff = diff * (np.float32(1 / stdev))
        inp.append(diff)
    print()

    inp = np.array(inp)

    return inp

try:
    model = keras.models.load_model("./weights.h5py")
    print("model loaded successfully!")
except:
    print("no model found...")

with open(os.path.join(test_inp, "meanstdev.txt")) as meanstdev:
    mean, stdev = map(float, meanstdev.readlines()[0:2])

count = 0
for j in range(1):
    inp = get_batch(test_inp, count * 128, length=128)
    preds = model.predict(inp, batch_size=16)
    for pred in preds:
        print((pred * stdev) + mean)
    count += 1
