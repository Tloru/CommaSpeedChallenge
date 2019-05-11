import argparse
import os

import model

parser = argparse.ArgumentParser()
command = parser.add_mutually_exclusive_group()
command.add_argument("process", action="store_true")
command.add_argument("train", action="store_true")
command.add_argument("test", action="store_true")
parser.add_argument("-i", default=None, help="path to input folder/file")
parser.add_argument("-o", default=None, help="path to output folder/file")

args = parser.parse_args()

if args.process:
    os.path.isfile()
