import torch
import torch.nn.functional as F
import torch.nn as nn

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear

    def forward(self, x):

model_factory = {
    'cnn': CNN
}

def save_model(model):
    from torch import save
    from os import path
    for n,  m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
        raise ValueError("model type '%s' not supported!" % str(type(model)))

def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model),
                           map_location='cpu'), strict=False)
    return r
