import torch
from models import save_model, CNN
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


def train(args):
    train_logger, val_logger = None, None

    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CNN()

    train_data = load_data(TRAIN_PATH)
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    loss = nn.CrossEntropyLoss()

    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        accuracy = []

        print("epoch: " + str(epoch))
        for img, label in train_data:
            # print("image is ", img)
            # print("label is ", label)
            # img, label = image.to(device), label.to(device)

            logit = model(img)
            label = label.long()
            # print("logit is ", logit)
            # print("logit type is ", type(logit))
            # print("label is ", label)
            # print("label type is ", type(label))
            loss_value = loss(logit, label)
            print("loss is ", loss_value)
            # if train_logger is not None:
            #     train_logger.add_scalar('loss', loss_value, global_step)
            # accuracy.append((accuracy.detach().cpu().numpy()))
            print("loss is ", loss_value)

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()
            global_step += 1

        # avg_acc = sum(accuracy) / len(accuracy)

        # if train_logger:
        #     train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()

    save_model(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)
