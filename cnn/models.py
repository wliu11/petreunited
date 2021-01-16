import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        #
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout()
        self.batchnorm = nn.BatchNorm2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128*12*12, out_features=120),
            nn.ReLU(),
            # nn.Linear(in_features=4096, out_features=4096),
            # nn.ReLU(),
            # nn.Linear(in_features=4096, out_features=120)
        )

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        # print("x shape ", x.shape)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x


if __name__ == '__main__':
    cnn = CNN()
    tensor = torch.zeros(128, 3, 100, 100)
    conv1 = cnn(tensor)


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNN):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNN()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
