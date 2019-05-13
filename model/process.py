from . import *

import os
import statistics as stat

def process(args):
    assert len(args) == 2
    inp = args[0]
    out = args[1]

    assert os.path.splitext(inp)[1] == ".txt"
    assert os.path.splitext(out)[1] == ".txt"

    # open the inp file and read all the lines
    with open(inp, "r") as text:
        lines = text.readlines()

    # convert the lines to numbers
    lines = list(map(float, lines))

    # calculate the mean and stdev of the dataset
    mean = stat.mean(lines)
    stdev = stat.stdev(lines)

    # normalize all the data in the dataset
    # add the normalization info the the head of the file
    # convert the numbers back to strings
    lines = list(map(lambda x: (x - mean) / stdev, lines))
    lines = [mean, stdev, ""] + lines
    lines = list(map(str, lines))

    # write the new data to the output file
    with open(out, "w") as text:
        text.write("\n".join(lines))
