import torch.nn as nn
from utils import *

class Recurrent(nn.Module):
  def __init__(self, args):
    super(Recurrent, self).__init__()
    self.args = args
    self.layers = nn.ModuleList()
    self.layers.append(nn.LSTM(args.input_size, args.hidden_size, args.hidden_layers))
    self.layers.append(nn.Linear(args.hidden_size, args.output_size))
    self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.SGD(self.parameters(), lr=args.learning_rate)
    return

  def forward(self, x):
    for layer in self.layers:
      x, _ = layer(x)
    return x

  def mask_grad(self):
    for layer in self.layers:
      if hasattr(layer, 'weight'):
        layer.weight.grad *= self.args.L1
        layer.weight.grad += self.args.L2 * layer.weight
    return

  def predict(self, x):
    self.eval()
    with torch.no_grad():
      return self(x).argmax(dim=1)

  def accuracy(self, x, y):
    return (self.predict(x) == y).float().mean()

  def energy(self, x, y):
    self.eval()
    with torch.no_grad():
      return self.criterion(self(x), y)

  def active(self, x):
    self.eval()
    with torch.no_grad():
      return self(x)

  def save(self, path):
    torch.save(self.state_dict(), path)
    return

  def load(self, path):
    self.load_state_dict(torch.load(path))
    return