from experiments.mnistnet import MNISTNET
from experiments.resnet import RESNET

experiments = [
    {"experiment": MNISTNET, "batch_size": 100, "epochs":"5"},
    {"experiment": RESNET, "batch_size": 100, "epochs":"10"},
]
