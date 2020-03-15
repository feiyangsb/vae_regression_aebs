"""
@author Feiyang Cai
@email feiyang.cai@vanderbilt.edu
@create date 2020-02-21 19:57:10
@modify date 2020-02-21 19:57:10
@desc This is the data loader for the AEBS data set 
"""

from torch.utils.data import Dataset
import csv
import os
from sklearn.model_selection import train_test_split
from PIL import Image
from torchvision import transforms
import numpy as np

class CarlaAEBSDataset(Dataset):
    def __init__(self, data_dir, split="train"):
        self.image_path_list, self.distance_list = self.build_data_list(data_dir)
        self.split = split
        self.image_path_list_train, self.image_path_list_calibration, self.distance_list_train, self.distance_list_calibration = \
                                            train_test_split(self.image_path_list, self.distance_list, test_size=0.2, random_state=42)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
    def build_data_list(self, data_dir):
        image_path_list = []
        distance_list = []
        for setting_dir in os.listdir(data_dir):
            for episode in os.listdir(os.path.join(data_dir, setting_dir)):
                csv_file_path = os.path.join(data_dir, setting_dir, episode, "label.csv")
                with open(csv_file_path) as csv_file:
                    csv_reader = csv.reader(csv_file, delimiter=',')
                    for index, row in enumerate(csv_reader):
                        image_path_list.append(os.path.join(data_dir, setting_dir, episode, str(index)+".png"))
                        distance_list.append(float(row[1]))
                        
        return image_path_list, distance_list
    
    def __len__(self):
        if self.split == "train":
            return len(self.distance_list_train)
        if self.split == "calibration":
            return len(self.distance_list_calibration)
    
    def __getitem__(self, idx):
        if self.split == "train":
            image_path_list = self.image_path_list_train
            distance_list = self.distance_list_train
        if self.split == "calibration":
            image_path_list = self.image_path_list_calibration
            distance_list = self.distance_list_calibration
        
        image = Image.open(image_path_list[idx]).convert("RGB")
        image = self.transform(image)
        distance = np.array([distance_list[idx]/120.0]).astype(np.float32)
        return image, distance