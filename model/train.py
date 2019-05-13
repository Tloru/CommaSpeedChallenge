from . import *

import cv2

def train(paths):

    BATCH_SIZE = 128
    

    # input.mp4, processed.txt
    assert len(args) == 2

    inp = cv2.VideoCapture()




    success, frame = inp.read()

    extractor = Extractor()
