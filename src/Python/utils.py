import os
import torch
from torchvision import datasets

DATA_PATH = "../../data/Python"
os.makedirs(DATA_PATH, exist_ok=True)

ERF_STR = "erf"
LINEAR_STR = "linear"
RELU_STR = "relu"
SIGMOID_STR = "sigmoid"
TANH_STR = "tanh"

ACT_FUNC = {
  ERF_STR : torch.erf,
  LINEAR_STR : lambda x: x,
  RELU_STR : torch.relu,
  SIGMOID_STR : torch.sigmoid,
  TANH_STR : torch.tanh
}

CIFAR10_STR = "CIFAR10"
CIFAR100_STR = "CIFAR100"
FASHIONMNIST_STR = "FashionMNIST"
MNIST_STR = "MNIST"
MNIST10_STR = "MNIST10"
DATASET = {
  # Name           : [Dataset, Normalize, Resize, Grayscale]
  CIFAR10_STR      : [datasets.CIFAR10, 10, ((0.4734,), (0.2393,)), None,
                      True],
  CIFAR100_STR     : [datasets.CIFAR100, 100, ((0.4782,), (0.2499,)), None,
                      True],
  FASHIONMNIST_STR : [datasets.FashionMNIST, 28, ((0.2860,), (0.3530,)), None,
                      False],
  MNIST_STR        : [datasets.MNIST, 10, ((0.1307,), (0.3081,)), None, False],
  MNIST10_STR      : [datasets.MNIST, 10, ((0.1307,), (0.2873,)), 10, False]
}


MSE_STR = "mse"
CE_STR = "ce"

LOSS = {
  MSE_STR : torch.nn.MSELoss(reduction='mean'),
  CE_STR : {
    1 : torch.nn.BCEWithLogitsLoss(reduction='mean')
  }
}

ADAM_STR = "Adam"
SGD_STR = "SGD"
RMS_STR = "RMS"

OPTIMIZER = {
  ADAM_STR : torch.optim.Adam,
  SGD_STR  : torch.optim.SGD,
  RMS_STR  : torch.optim.RMSprop

}

DEVICE = None
kwargs = {}

def init():
  global DEVICE
  global kwargs
  USE_CUDA = torch.cuda.is_available()
  DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
  kwargs = {"num_workers": 8, "pin_memory": True} if USE_CUDA else {}
  print(DEVICE)
