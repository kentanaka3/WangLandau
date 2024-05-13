import argparse
import numpy as np
import torchvision

from utils import *
from Supervised import *
from Unsupervised import *
from Reinforcement import *

init()

MULTILAYER_PERCEPTRON_STR = "MultiLayerPerceptron"
FEEDFOWARD_STR            = "FeedForward"
RECURRENT_STR             = "Recurrent"
CONVOLUTIONAL_STR         = "Convolutional"
TRANSFORMER_STR           = "Transformer"
AUTOENCODER_STR           = "AutoEncoder"

def parse_arguments():
  parser = argparse.ArgumentParser(description="ML")
  # TODO: Group arguments by Learning type
  #       (Supervised, Unsupervised, Reinforcement)
  parser.add_argument('-A', "--activation", default=TANH_STR, required=False,
                      choices=ACT_FUNC.keys(), help="Activation function")
  parser.add_argument('-B', "--batch", default=[10, 10], required=False,
                      nargs=2,
                      help="Batch sizes for Train and Test, respectively")
  parser.add_argument('-D', "--dataset", default=MNIST_STR, required=False,
                      choices=DATASET.keys(),
                      help="Select the standard Dataset to analyze")
  parser.add_argument("-E", "--epochs", default=1000, type=int, required=False)
  parser.add_argument("-K", "--K-hidden-units", default=[11], type=int,
                      nargs='+', required=False, help="Number of hidden units")
  parser.add_argument('-L', "--loss", default=MSE_STR, choices=LOSS.keys())
  parser.add_argument('-N', "--network", default=MULTILAYER_PERCEPTRON_STR,
                      choices=NETWORK.keys())
  parser.add_argument('-O', "--optimizer", default=SGD_STR, required=False,
                      choices=OPTIMIZER.keys())
  parser.add_argument('-T', "--task", default="NLGP",
                      choices=["NLGP", "R", "R-Y", "R-FF-T", "R-FF-T-Y",
                               "R-RNN-T", "R-RNN-T-X", "R-RNN-T-Y"])
  parser.add_argument("--L1", default=0., type=float, required=False,
                      help="L1 weight regularization")
  parser.add_argument("--L2", default=0., type=float, required=False,
                      help="L2 weight regularization")
  parser.add_argument("--learning-rate", default=0.05, type=float, nargs=1,
                      required=False, help="Learning Rate")
  parser.add_argument("--torus", default=True, action='store_true',
                      required=False,
                      help="Boolean whether to use a torus topology")
  parser.add_argument('-v', "--verbose", action='store_true', required=False,
                      default=False, help='Enable verbose mode')
  return parser.parse_args()

def setUp(args):
  dataset = DATASET[args.dataset]
  TRANSFORM = [torchvision.transforms.ToTensor()]
  # Normalize
  TRANSFORM += [torchvision.transforms.Normalize(*dataset[2])]
  if dataset[3] is not None:
    TRANSFORM += [torchvision.transforms.Resize(dataset[3])]
  if dataset[4]: TRANSFORM += [torchvision.transforms.Grayscale()]
  TRANSFORM = torchvision.transforms.Compose(TRANSFORM)
  # Train
  TrainSet = dataset[0](DATA_PATH, train=True, download=True,
                        transform=TRANSFORM)
  TrainLoader = torch.utils.data.DataLoader(TrainSet, batch_size=args.batch[0],
                                            shuffle=True, **kwargs)
  # Test
  TestSet = dataset[0](DATA_PATH, train=False, download=True,
                       transform=TRANSFORM)
  TestLoader = torch.utils.data.DataLoader(TestSet, batch_size=args.batch[1],
                                           shuffle=True, **kwargs)
  # Model
  Model = NETWORK[args.network](args)

  return TrainLoader, TestLoader, Model

def modelTrain(args, Model, TrainLoader, TestLoader):
  for ep in range(1, args.epochs + 2):
    Model.train()
    loss_tr_ep, err_tr_ep, acc_tr_ep, active_tr_ep, energy_tr_ep = \
      0., 0., 0., 0., 0.
    for batch_idx, (data, target) in enumerate(TrainLoader):
      XS, output = Model(data)
      X = XS[-1]
      loss = Model.criterion(output, target)
      Model.optimizer.zero_grad()
      Model.mask_grad()
      Model.optimizer.step()
  return

NETWORK = {
  MULTILAYER_PERCEPTRON_STR                   : MultiLayerPerceptron,
  FEEDFOWARD_STR                              : FreeForward,
  RECURRENT_STR                               : Recurrent,
  CONVOLUTIONAL_STR                           : Convolutional,
  TRANSFORMER_STR                             : Transformer,
  AUTOENCODER_STR                             : AutoEncoder,
  AUTOENCODER_STR + MULTILAYER_PERCEPTRON_STR : AutoEncoder,
  AUTOENCODER_STR + FEEDFOWARD_STR            : AutoEncoderFreeForward,
  AUTOENCODER_STR + CONVOLUTIONAL_STR         : AutoEncoderConvolutional,
  AUTOENCODER_STR + RECURRENT_STR             : AutoEncoderRecurrent,
  AUTOENCODER_STR + TRANSFORMER_STR           : AutoEncoderTransformer
}