
iterations = [
    {"batch": 50, "epochs": range(1, 11)},
    {"batch": 100, "epochs": 2},
    {"batch": 100, "epochs": 3},
]

def setup(dataset, **kwargs):
    batch=kwargs["batch"]
    epochs=kwargs["epochs"]

def teardown():
    pass