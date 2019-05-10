import argparse
import os
import shutil
import sys
import numpy as np

import cv2

parser = argparse.ArgumentParser()
parser.add_argument("input_path", type=str, help="file path of video file to preprocess")
parser.add_argument("output_path", type=str, help="folder path of place to save frame sequence")

args = parser.parse_args()
input_path  = os.path.abspath(args.input_path)
output_path = os.path.abspath(args.output_path)

def print_progress(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + ' ' * (bar_length - filled_length)

    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),

    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()

def read_frames_sequel(input_path, output_path, stack):
    vidcap = cv2.VideoCapture(input_path)
    total = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    average = []
    for j in range(stack):
        average.append(vidcap.read()[1])

    shape = (
        int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        3
    )

    success = True
    count = 0

    while success:
        success, frame = vidcap.read()

        average.pop(0)
        average.append(frame)

        diff = np.zeros(shape, np.uint8)

        for frame in average:
            diff = cv2.absdiff(frame, diff)

        diff = diff[:][20:320]
        diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        cv2.imwrite(os.path.join(output_path, ("frame%6d.png" % count).replace(" ", "0")), diff)
        print_progress(count, total, prefix="frame: {}/{}".format(count + 1, total), bar_length=32)

        count += 1

if os.path.isdir(output_path):
    shutil.rmtree(output_path)
os.makedirs(output_path)

read_frames_sequel(input_path, output_path, 10)
