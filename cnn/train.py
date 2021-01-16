import torch
from torch.autograd import Variable

from models import save_model, CNN
from utils import load_data
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
import torch.nn as nn
import torch.utils.tensorboard as tb
from os import path
import argparse

num_epochs = 35
lr = 1e-3

TRAIN_PATH = 'lists/train_list.mat'
VALIDATION_PATH = 'lists/test_list.mat'


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

#
# # https://github.com/asifdahir/FourShapes_CNN/blob/af2e140c1233d991a4701fcc73cd6898cd3badbe/shapes_cnn.py
# def fit(epoch, model, dataloader, phase='train'):
#     print("epoch # ", epoch)
#     device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#
#     if phase == 'train':
#         model.train()
#     if phase == 'validation':
#         model.eval()
#
#     running_loss = 0.0
#     running_correct = 0
#
#     for data, target in dataloader:
#         data, target = data.to(device), target.to(device)
#         logit = model(data)
#         loss_val = loss_fn(logit, target.long())
#         acc_val = accuracy(logit, target)
#
#         optimizer.zero_grad()
#         loss_val.backward()
#         optimizer.step()
#
#         # if phase == 'train':
#         #     optimizer.zero_grad()
#         # output = model(data)
#         # loss = loss_fn(output, target.long())
#         # _, preds = torch.max(output.data, dim=1)
#         # # running_loss += loss.detach()
#         # running_correct += preds.eq(target.data.view_as(preds)).float().sum()
#         # # if phase == 'train':
#         # optimizer.step()
#
#     loss = running_loss / len(dataloader.dataset)
#     accuracy = 100. * running_correct / len(dataloader.dataset)
#
#     print(f'{phase} loss is {loss} and {phase} accuracy is {running_correct}/{len(dataloader.dataset)} = {accuracy}')
#     return loss, accuracy


def train():
    from os import path
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))

    """
    Your code here, modify your HW1 code

    """
    import torch

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = CNN().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    loss = torch.nn.CrossEntropyLoss()

    train_data = load_data('lists/train_list.mat')
    valid_data = load_data('lists/test_list.mat')

    global_step = 0
    for epoch in range(args.num_epoch):
        print("epoch ", epoch)
        model.train()
        acc_vals = []
        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            logit = model(img)
            loss_val = loss(logit, label.long())
            acc_val = accuracy(logit, label)

            if train_logger is not None:
                train_logger.add_scalar('loss', loss_val, global_step)
            acc_vals.append(acc_val.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            global_step += 1
        avg_acc = sum(acc_vals) / len(acc_vals)

        if train_logger:
            train_logger.add_scalar('accuracy', avg_acc, global_step)

        model.eval()
        acc_vals = []
        for img, label in valid_data:
            img, label = img.to(device), label.to(device)
            acc_vals.append(accuracy(model(img), label).detach().cpu().numpy())
        avg_vacc = sum(acc_vals) / len(acc_vals)

        if valid_logger:
            valid_logger.add_scalar('accuracy', avg_vacc, global_step)

        if valid_logger is None or train_logger is None:
            print('epoch %-3d \t acc = %0.3f \t val acc = %0.3f' % (epoch, avg_acc, avg_vacc))
        save_model(model)
    save_model(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    parser.add_argument('-n', '--num_epoch', type=int, default=20)
    # parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)
    args = parser.parse_args()
    train()

