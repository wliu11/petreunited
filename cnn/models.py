import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(in_features=64*56*56, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=120)

    def forward(self, x):
        # print("tensor shape is at first ", x.shape)
        x = self.relu(self.conv1(x))
        # print("tensor shape after conv1 ", x.shape)

        x = self.pool(x)
        # print("after the first max pool, x shape is ", x.shape)
        x = self.relu(self.conv2(x))
        # print("tensor shape after conv2 ", x.shape)

        x = self.pool(x)
        x = x.view(-1, 64*56*56)
        # print("after the second max pool, x shape is ", x.shape)
        # x = x.view(-1, 12800)
        x = self.relu(self.fc1(x))
        # x = F.softmax(self.fc1(x), dim=1)

        # return self.fc2(x)
        # print("after fc1, x shape is ", x.shape)
        # x = self.fc2(x)
        # print("after fc2 x shape is ", x.shape)
        return x


if __name__ == '__main__':
    cnn = CNN()
    tensor = torch.zeros(128, 3, 224, 224)
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
