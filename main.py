print("starting...")

import argparse
import os

import model

parser = argparse.ArgumentParser()
parser.add_argument("command", type=str, help="command to run")
command, paths = parser.parse_known_args()
command_name = command.command

commands = {
    "process": model.process,
    "train":   model.train,
    "test":    model.test
}

paths = list(map(os.path.abspath, paths))
command = commands[command_name]

print("running '{}' command...".format(command_name))
command(paths)
print("task successful!")
