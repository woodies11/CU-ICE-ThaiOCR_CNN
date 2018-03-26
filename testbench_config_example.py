from experiments.mnistnet import MNISTNET
from experiments.resnet import RESNET

"""
Rename this file to: testbench_config.py
"""

# experiment must contain an Experiment class (thhe module itself, not an instance)
# batch_size can either be an int or a range() object
# epochs must be an int
experiments = [
    {"experiment": MNISTNET, "batch_size": range(25, 100, 25), "epochs":20},
    {"experiment": RESNET, "batch_size": 100, "epochs":10},
]
