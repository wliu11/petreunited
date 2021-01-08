import torch
from models import model_factory, save_model, CNN
from utils import load_data
import torch.optim as optim
from torchvision import transforms
import torch.nn as nn
import torch.utils.tensorboard as tb
from os import path
import argparse

num_epochs = 35
lr = 1e-3

TRAIN_PATH = 'lists/train_list.mat'

transforms = {
    transforms.Compose([
        transforms.RandomCrop(200),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
}


def train(args):
    train_logger, val_logger = None, None

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CNN().to(device)

    train_data = load_data(TRAIN_PATH)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss = nn.CrossEntropyLoss()

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        accuracy = []

        print("epoch: " + str(epoch))
        for image, label in train_data:
            img, label = image.to(device), label.to(device)
            logit = model(img)
            loss_value = loss(logit, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_value, global_step)
            accuracy.append((accuracy.detach().cpu().numpy()))

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            global_step += 1

        avg_acc = sum(accuracy) / len(accuracy)

        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()

    save_model(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
