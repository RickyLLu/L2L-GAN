"""
The dataloader for different datasets, currently only VGG-Face is considered
"""
VGG_PATH = '/store/vgg_face_dataset/files/'
TRAIN_LIST = '/store/vgg_face_dataset/train_list.txt'
VAL_LIST = '/store/vgg_face_dataset/val_list.txt'
TEST_LIST = '/store/vgg_face_dataset/test_list.txt'

import torch
import os
import numpy as np
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import warnings


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preprocess_vggface():
    files = os.listdir(VGG_PATH)
    paths = []
    for file in files:
        if file[-4:] != '.txt':
            paths.append(os.path.join(VGG_PATH, file))

    train_path = paths[:-10]
    val_path = paths[-10:-5]
    test_path = paths[-5:]
    train_list, val_list, test_list = [], [], []
    for p in train_path:
        imgs = os.listdir(p)
        for img in imgs:
            if img[-4:] != '.err':
                train_list.append(os.path.join(p, img))

    for p in val_path:
        imgs = os.listdir(p)
        for img in imgs:
            if img[-4:] != '.err':
                val_list.append(os.path.join(p, img))

    for p in test_path:
        imgs = os.listdir(p)
        for img in imgs:
            if img[-4:] != '.err':
                test_list.append(os.path.join(p, img))

    with open(TRAIN_LIST, 'w') as f:
        for p in train_list:
            f.write(p+'\r\n')
    with open(VAL_LIST, 'w') as f:
        for p in val_list:
            f.write(p+'\r\n')
    with open(TEST_LIST, 'w') as f:
        for p in test_list:
            f.write(p+'\r\n')

class VggFace_Dataset(Dataset):
    def __init__(self, txt_file, transform=transforms.ToTensor()):
        self.images = open(txt_file, 'r').readlines()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        lbl = img_path.split('/')[4]
        img_path = img_path[:-1]
        img = Image.open(img_path)
        image = img.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, lbl

class DaganDatasetVgg(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, x1_examples, x2_examples, transform=None):
        assert len(x1_examples) == len(x2_examples)
        self.x1_examples = x1_examples
        self.x2_examples = x2_examples
        self.totensor = transforms.ToTensor()
        self.transform = transform

    def __len__(self):
        return len(self.x1_examples)

    def __getitem__(self, idx):
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            return self.transform(self.x1_examples[idx]), self.transform(
                self.x2_examples[idx]
            )
        """
        img_path1 = self.x1_examples[idx]
        img_path2 = self.x2_examples[idx]
        img1 = Image.open(img_path1)
        image1 = img1.convert('RGB')
        img2 = Image.open(img_path2)
        image2 = img2.convert('RGB')
        image1 = self.totensor(image1)
        image2 = self.totensor(image2)
        if self.transform:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        return image1, image2


def create_dagan_vgg_dataloader(transform, batch_size):
    files = os.listdir(VGG_PATH)
    paths = []
    for file in files:
        if file[-4:] != '.txt':
            paths.append(os.path.join(VGG_PATH, file))

    train_path = paths[:-10]
    num_classes = len(train_path)
    train_x1 = []
    train_x2 = []
    for p in train_path:
        train_list = []
        imgs = os.listdir(p)
        for img in imgs:
            if img[-4:] != '.err':
                train_list.append(os.path.join(p, img))

        x2_data_path = train_list
        np.random.shuffle(x2_data_path)
        train_x1.extend(train_list)
        train_x2.extend(x2_data_path)

    train_dataset = DaganDatasetVgg(train_x1, train_x2, transform)
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)

if __name__=='__main__':
    # preprocess_vggface()
    kwargs = {'num_workers': 1, 'pin_memory': True}
    vgg_data = VggFace_Dataset(TEST_LIST)
    test_loader = torch.utils.data.DataLoader(vgg_data, batch_size=4, shuffle=True, **kwargs)
    for img, label in test_loader:
        print(img.shape, label)