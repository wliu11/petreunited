import os
import xml.etree.cElementTree as ET

from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

transformation = {
    'train': transforms.Compose([
        transforms.Resize([100, 100]),
        # transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()]),
    'test': transforms.Compose([
        # transforms.Resize([224, 224]),
        transforms.ToTensor()])
}


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
            img_name = self.image_names[idx][0][0]
            img_path = os.path.join("Images" + "/" + img_name)
            image = Image.open(img_path).convert('RGB')
            image.load()
            image = image.crop(bounding_box(img_name))
            image = self.transform(image)
            self.images.append(image)

    def __len__(self):
        return len(self.image_names)

    # Lazy processing of each image into tensor, along with its label
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx][0] - 1
        return image, label

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


def load_data(dataset_path, phase='train', num_workers=0, batch_size=128):
    dataset = DogDataset(dataset_path, transform=transformation[phase])
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
