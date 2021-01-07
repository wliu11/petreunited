from scipy.io import loadmat
import xml.etree.cElementTree as ET
import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import matplotlib.pyplot as plt


class DogDataset(Dataset):

    def __init__(self, train_dir, transform=None):

        data = loadmat(train_dir)

        # List of labels
        self.labels = [[label for label in element] for element in data['labels']]

        # List of image file names
        self.images = [[image for image in element] for element in data['file_list']]

        # List of annotations files- contains information such as the bounding box
        self.annotations = [[annot for annot in element] for element in data['annotation_list']]
        self.transform = transform

        self.data = []

        for idx in range(self.__len__()):
            img_name = self.images[idx][0][0]
            img_path = os.path.join("Images" + "/" + img_name)
            image = Image.open(img_path)

            crop = bounding_box(img_name)
            image = image.crop(crop)

            image.load()
            label = self.labels[idx][0]
            self.data.append((image, label))

    def __len__(self):
        return len(self.images)

    # Lazy processing of each image into tensor, along with its label
    def __getitem__(self, idx):
        data = self.data[idx]
        if self.transform:
            data = self.transform(data)
        return data

    def show_image(self, idx):
        image, label = self.__getitem__(idx)
        image.show()


def bounding_box(img_name):
    path = "Annotation/" + img_name[:-4]
    tree = ET.ElementTree(file=path)
    root = tree.getroot()

    for elem in root.findall('object/bndbox'):
        xmin = int(elem.find('xmin').text)
        ymin = int(elem.find('ymin').text)
        xmax = int(elem.find('xmax').text)
        ymax = int(elem.find('ymax').text)

    return xmin, ymin, xmax, ymax


if __name__ == '__main__':
    dog_dataset = DogDataset(train_dir='lists/train_list.mat')
    # for i in range(len(dog_dataset)):
    #     sample = dog_dataset[i]
    dog_dataset.show_image(400)


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = DogDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()