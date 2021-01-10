from scipy.io import loadmat
import xml.etree.cElementTree as ET
import pandas as pd
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt


class DogDataset(Dataset):

    def __init__(self, train_dir, transform):

        data = loadmat(train_dir)

        # List of labels
        self.labels = [[label for label in element] for element in data['labels']]

        # List of image file names
        self.image_names = [[image for image in element] for element in data['file_list']]

        # List of annotations files- contains information such as the bounding box
        self.annotations = [[annot for annot in element] for element in data['annotation_list']]

        self.transform = transform

        self.images = []

        for idx in range(self.__len__()):
            to_tensor = transforms.ToTensor()
            resize = transforms.Resize(256)
            random_crop = transforms.RandomCrop(224)
            img_name = self.image_names[idx][0][0]
            img_path = os.path.join("Images" + "/" + img_name)
            image = Image.open(img_path)
            image.load()
            crop = bounding_box(img_name)
            image = image.crop(crop)
            resized = resize(image)
            cropped = random_crop(resized)
            self.images.append(to_tensor(cropped))

    def __len__(self):
        print(len(self.image_names))
        return len(self.image_names)

    # Lazy processing of each image into tensor, along with its label
    def __getitem__(self, idx):
        image = self.images[idx]
        # print("image is first ", image)
        label = self.labels[idx][0]
        # print("label is ", label)
        # image = self.transform(image)
        # print("after transform image is a ", type(image))
        ret = (image, label)
        # print("type of return is ", type(ret))
        return ret

    def show_image(self, idx):
        image, label = self.__getitem__(idx)
        image.show()


def bounding_box(img_name):

    # Splice out the ".jpg" marker from the end of each image name to get the annotation file
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
    dog_dataset.show_image(400)


def load_data(dataset_path, num_workers=0, batch_size=128):
    transformation = {
        transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }

    dataset = DogDataset(dataset_path, transform=transformation)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()