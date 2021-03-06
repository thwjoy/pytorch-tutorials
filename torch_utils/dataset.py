import torch
import torch.utils.data
import os
from skimage import io
import pandas as pd
import numpy as np


class mnistmTrainingDataset(torch.utils.data.Dataset):

    def __init__(self, text_file):
        """
        Args:
            text_file(string): path to text file
        """
        self.name_frame = pd.read_csv(text_file, sep=",", usecols=range(1))
        self.label_frame = pd.read_csv(text_file, sep=",", usecols=range(1, 2))

    def __len__(self):
        return len(self.name_frame)

    def __getitem__(self, idx):
        img_name = self.name_frame.iloc[idx, 0]
        image = io.imread(img_name).astype(float)
        image *= 1.0/image.max()
        image = np.expand_dims(image, axis=0)
        labels = self.label_frame.iloc[idx, 0]
        sample = {'image': image, 'labels': labels}
        return sample
