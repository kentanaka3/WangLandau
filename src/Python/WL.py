from ML import *

def main(args):
  TrainLoader, TestLoader, Model = setUp(args)
  inter_model = modelTrain(args, Model, TrainLoader, TestLoader)
  return

if __name__ == "__main__":
  args = parse_arguments()
  main(args)