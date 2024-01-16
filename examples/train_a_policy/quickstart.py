"""
This file contains an example of training a reinforcement learning agent with or without a shield.
"""

import os
from pls.workflows.execute_workflow import train, test


# Use the pretrained model
if __name__ == "__main__":
    # current working directory
    cwd = os.path.join(os.path.dirname(__file__))

    # location of the config file
    config_file = os.path.join(cwd, "carracing/no_shield/seed1/config.json")

    # trains the reinforcement agent using the parameters in the config file
    train(config_file)
