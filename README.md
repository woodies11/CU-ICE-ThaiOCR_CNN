# greenmotherboard
A totally normal green motherboard


# Setting up the Environment

From [Keras MNIST Tutorial](https://elitedatascience.com/keras-tutorial-deep-learning-in-python)

## Set Up Python Environment

Create a Virtual Environment using Conda.
```bash
conda create -n ENVNAME
``` 

Activate the environment.
```bash
source activate ENVNAME
```


## Set Up ML Frameworks

You can either use Theano or Tensorflow. Config the backend choice in `~/.keras/keras.json`

### Theano
```bash
conda install numpy scipy mkl nose sphinx pydot-ng
```

config in `~/.theanorc`
Mac OS X 10.13 have some problems with CUDA on my machine (GT750m) so I will use CPU only for now.

### Keras
```bash
pip install keras
```


# Handy Snippets

## Python

### Anaconda

#### Create New Environment
```bash
conda create -n ENVNAME
```

#### List all Conda Environment
```bash
conda env list
```

