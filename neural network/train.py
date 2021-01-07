from .utils import load_data
import torch.optim as optim

num_epochs = 35
lr = 1e-3

def train(args):
    model = model_factory[args.model]()
