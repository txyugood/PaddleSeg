import os
import sys

import cv2
from PIL import Image

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.."))

import random
import h5py
import numpy as np

from medicalseg.datasets import MedicalDataset
from medicalseg.cvlibs import manager
from medicalseg.transforms import Compose


@manager.DATASETS.add_component
class Synapse(MedicalDataset):
    def __init__(self,
                 dataset_root,
                 mode,
                 num_classes,
                 result_dir,
                 transforms=None):
        if isinstance(transforms, list):
            transforms = Compose(transforms)
        self.transforms = transforms
        self.mode = mode
        self.sample_list = open(
            os.path.join(dataset_root, self.mode + '.txt')).readlines()
        self.dataset_root = dataset_root
        self.num_classes = num_classes
        self.result_dir = result_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        sample = self.sample_list[idx].strip('\n')
        image_path, label_path = sample.split(' ')

        image = cv2.imread(os.path.join(self.dataset_root, image_path), 0)
        label = np.array(
            Image.open(os.path.join(self.dataset_root, label_path)))
        image = image[np.newaxis, :, :].astype('float32') / 255.0
        label = label[np.newaxis, :, :]
        if self.transforms:
            image, label = self.transforms(im=image, label=label)

        return image.astype('float32'), label.astype('int64'), self.sample_list[
            idx].strip('\n').split(" ")[0].split('/')[-1].split('_')[0]
