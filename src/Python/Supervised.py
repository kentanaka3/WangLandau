from utils import *

# Networks
class MultiLayerPerceptron(torch.nn.Module):
  def __init__(self, args):
    super(MultiLayerPerceptron, self).__init__()
    self.layers = []
    self.activation = ACT_FUNC[args.activation]
    self.criterion = LOSS[args.loss]
    self.optimizer = OPTIMIZER[args.optimizer](self.parameters(),
                                               lr=args.learning_rate)
    self.K = args.K_hidden_units
    self.L1 = args.L1
    self.L2 = args.L2
    self.torus = args.torus
    self.verbose = args.verbose
    self.init_layers()

  def init_layers(self):
    self.layers = [torch.nn.Linear(784, self.K[0])]
    for i in range(1, len(self.K)):
      self.layers += [torch.nn.Linear(self.K[i - 1], self.K[i])]
    self.layers += [torch.nn.Linear(self.K[-1], 10)]
    self.layers = torch.nn.ModuleList(self.layers)

  def mask_grad(self):
    for layer in self.layers:
      if self.L1 > 0.:
        layer.weight.grad += self.L1 * torch.sign(layer.weight)
      if self.L2 > 0.:
        layer.weight.grad += self.L2 * 2. * layer.weight

  def forward(self, x):
    XS = [x.view(-1, 784)]
    for i, layer in enumerate(self.layers):
      XS += [self.activation(layer(XS[-1]))]
    return XS, XS[-1]

class FreeForward(torch.nn.Module):
  def __init__(self, args):
    super(FreeForward, self).__init__()
    self.layers = []
    self.activation = ACT_FUNC[args.activation]
    self.criterion = LOSS[args.loss]
    self.optimizer = OPTIMIZER[args.optimizer](self.parameters(),
                                               lr=args.learning_rate)
    self.K = args.K_hidden_units
    self.L1 = args.L1
    self.L2 = args.L2
    self.torus = args.torus
    self.verbose = args.verbose
    self.init_layers()

  def init_layers(self):
    self.layers = [torch.nn.Linear(784, self.K[0])]
    for i in range(1, len(self.K)):
      self.layers += [torch.nn.Linear(self.K[i - 1], self.K[i])]
    self.layers += [torch.nn.Linear(self.K[-1], 10)]
    self.layers = torch.nn.ModuleList(self.layers)

  def mask_grad(self):
    for layer in self.layers:
      if self.L1 > 0.:
        layer.weight.grad += self.L1 * torch.sign(layer.weight)
      if self.L2 > 0.:
        layer.weight.grad += self.L2 * 2. * layer.weight

  def forward(self, x):
    XS = [x.view(-1, 784)]
    for i, layer in enumerate(self.layers):
      XS += [self.activation(layer(XS[-1]))]
    return XS, XS[-1]

class Recurrent(torch.nn.Module):
  def __init__(self, args):
    super(Recurrent, self).__init__()
    self.layers = []
    self.activation = ACT_FUNC[args.activation]
    self.criterion = LOSS[args.loss]
    self.optimizer = OPTIMIZER[args.optimizer](self.parameters(),
                                               lr=args.learning_rate)
    self.K = args.K_hidden_units
    self.L1 = args.L1
    self.L2 = args.L2
    self.torus = args.torus
    self.verbose = args.verbose
    self.init_layers()

  def init_layers(self):
    self.layers = [torch.nn.Linear(784, self.K[0])]
    for i in range(1, len(self.K)):
      self.layers += [torch.nn.Linear(self.K[i - 1], self.K[i])]
    self.layers += [torch.nn.Linear(self.K[-1], 10)]
    self.layers = torch.nn.ModuleList(self.layers)

  def mask_grad(self):
    for layer in self.layers:
      if self.L1 > 0.:
        layer.weight.grad += self.L1 * torch.sign(layer.weight)
      if self.L2 > 0.:
        layer.weight.grad += self.L2 * 2. * layer.weight

  def forward(self, x):
    XS = [x.view(-1, 784)]
    for i, layer in enumerate(self.layers):
      XS += [self.activation(layer(XS[-1]))]
    return XS, XS[-1]

class Convolutional(torch.nn.Module):
  def __init__(self, args):
    super(Convolutional, self).__init__()
    self.layers = []
    self.activation = ACT_FUNC[args.activation]
    self.criterion = LOSS[args.loss]
    self.optimizer = OPTIMIZER[args.optimizer](self.parameters(),
                                               lr=args.learning_rate)
    self.K = args.K_hidden_units
    self.L1 = args.L1
    self.L2 = args.L2
    self.torus = args.torus
    self.verbose = args.verbose
    self.init_layers()

  def init_layers(self):
    self.layers = [torch.nn.Conv2d(1, self.K[0], 3, padding=1)]
    for i in range(1, len(self.K)):
      self.layers += [torch.nn.Conv2d(self.K[i - 1], self.K[i], 3, padding=1)]
    self.layers += [torch.nn.Linear(self.K[-1], 10)]
    self.layers = torch.nn.ModuleList(self.layers)

  def mask_grad(self):
    for layer in self.layers:
      if self.L1 > 0.:
        layer.weight.grad += self.L1 * torch.sign(layer.weight)
      if self.L2 > 0.:
        layer.weight.grad += self.L2 * 2. * layer.weight

  def forward(self, x):
    XS = [x.view(-1, 784)]
    for i, layer in enumerate(self.layers):
      XS += [self.activation(layer(XS[-1]))]
    return XS, XS[-1]

class Transformer(torch.nn.Module):
  def __init__(self, args):
    super(Transformer, self).__init__()
    self.layers = []
    self.activation = ACT_FUNC[args.activation]
    self.criterion = LOSS[args.loss]
    self.optimizer = OPTIMIZER[args.optimizer](self.parameters(),
                                               lr=args.learning_rate)
    self.K = args.K_hidden_units
    self.L1 = args.L1
    self.L2 = args.L2
    self.torus = args.torus
    self.verbose = args.verbose
    self.init_layers()

  def init_layers(self):
    self.layers = [torch.nn.Transformer(784, self.K[0])]
    for i in range(1, len(self.K)):
      self.layers += [torch.nn.Transformer(self.K[i - 1], self.K[i])]
    self.layers += [torch.nn.Linear(self.K[-1], 10)]
    self.layers = torch.nn.ModuleList(self.layers)

class AutoEncoder(torch.nn.Module):
  def __init__(self, args):
    super(AutoEncoder, self).__init__()
    self.layers = []
    self.activation = ACT_FUNC[args.activation]
    self.criterion = LOSS[args.loss]
    self.optimizer = OPTIMIZER[args.optimizer](self.parameters(),
                                               lr=args.learning_rate)
    self.K = args.K_hidden_units
    self.L1 = args.L1
    self.L2 = args.L2
    self.torus = args.torus
    self.verbose = args.verbose
    self.init_layers()

  def init_layers(self):
    self.layers = [torch.nn.Linear(784, self.K[0])]
    for i in range(1, len(self.K)):
      self.layers += [torch.nn.Linear(self.K[i - 1], self.K[i])]
    self.layers += [torch.nn.Linear(self.K[-1], 784)]
    self.layers = torch.nn.ModuleList(self.layers)

  def mask_grad(self):
    for layer in self.layers:
      if self.L1 > 0.:
        layer.weight.grad += self.L1 * torch.sign(layer.weight)
      if self.L2 > 0.:
        layer.weight.grad += self.L2 * 2. * layer.weight

  def forward(self, x):
    XS = [x.view(-1, 784)]
    for i, layer in enumerate(self.layers):
      XS += [self.activation(layer(XS[-1]))]
    return XS, XS[-1]

class AutoEncoderConvolutional(torch.nn.Module):
  def __init__(self, args):
    super(AutoEncoderConvolutional, self).__init__()
    self.layers = []
    self.activation = ACT_FUNC[args.activation]
    self.criterion = LOSS[args.loss]
    self.optimizer = OPTIMIZER[args.optimizer](self.parameters(),
                                               lr=args.learning_rate)
    self.K = args.K_hidden_units
    self.L1 = args.L1
    self.L2 = args.L2
    self.torus = args.torus
    self.verbose = args.verbose
    self.init_layers()

  def init_layers(self):
    self.layers = [torch.nn.Conv2d(1, self.K[0], 3, padding=1)]
    for i in range(1, len(self.K)):
      self.layers += [torch.nn.Conv2d(self.K[i - 1], self.K[i], 3, padding=1)]
    self.layers += [torch.nn.Conv2d(self.K[-1], 1, 3, padding=1)]
    self.layers = torch.nn.ModuleList(self.layers)

  def mask_grad(self):
    for layer in self.layers:
      if self.L1 > 0.:
        layer.weight.grad += self.L1 * torch.sign(layer.weight)
      if self.L2 > 0.:
        layer.weight.grad += self.L2 * 2. * layer.weight

  def forward(self, x):
    XS = [x.view(-1, 784)]
    for i, layer in enumerate(self.layers):
      XS += [self.activation(layer(XS[-1]))]
    return XS, XS[-1]

class AutoEncoderRecurrent(torch.nn.Module):
  def __init__(self, args):
    super(AutoEncoderRecurrent, self).__init__()
    self.layers = []
    self.activation = ACT_FUNC[args.activation]
    self.criterion = LOSS[args.loss]
    self.optimizer = OPTIMIZER[args.optimizer](self.parameters(),
                                               lr=args.learning_rate)
    self.K = args.K_hidden_units
    self.L1 = args.L1
    self.L2 = args.L2
    self.torus = args.torus
    self.verbose = args.verbose
    self.init_layers()

  def init_layers(self):
    self.layers = [torch.nn.Linear(784, self.K[0])]
    for i in range(1, len(self.K)):
      self.layers += [torch.nn.Linear(self.K[i - 1], self.K[i])]
    self.layers += [torch.nn.Linear(self.K[-1], 784)]
    self.layers = torch.nn.ModuleList(self.layers)

  def mask_grad(self):
    for layer in self.layers:
      if self.L1 > 0.:
        layer.weight.grad += self.L1 * torch.sign(layer.weight)
      if self.L2 > 0.:
        layer.weight.grad += self.L2 * 2. * layer.weight

  def forward(self, x):
    XS = [x.view(-1, 784)]
    for i, layer in enumerate(self.layers):
      XS += [self.activation(layer(XS[-1]))]
    return XS, XS[-1]

class AutoEncoderTransformer(torch.nn.Module):
  def __init__(self, args):
    super(AutoEncoderTransformer, self).__init__()
    self.layers = []
    self.activation = ACT_FUNC[args.activation]
    self.criterion = LOSS[args.loss]
    self.optimizer = OPTIMIZER[args.optimizer](self.parameters(),
                                               lr=args.learning_rate)
    self.K = args.K_hidden_units
    self.L1 = args.L1
    self.L2 = args.L2
    self.torus = args.torus
    self.verbose = args.verbose
    self.init_layers()

  def init_layers(self):
    self
    self.layers = [torch.nn.Transformer(784, self.K[0])]
    for i in range(1, len(self.K)):
      self.layers += [torch.nn.Transformer(self.K[i - 1], self.K[i])]
    self.layers += [torch.nn.Linear(self.K[-1], 784)]
    self.layers = torch.nn.ModuleList(self.layers)