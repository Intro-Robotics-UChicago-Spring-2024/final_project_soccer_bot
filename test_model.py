#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, random_split
from torchvision import transforms
from torchvision.transforms import v2
from os import walk
from os import listdir
import pandas as pd
import numpy as np
import torchvision.models as models

from PIL import Image

def merge_data(image_data, action_data):
    dataset = []

    for index, image in enumerate(image_data):
        dataset.append((image, action_data[index]))

    return dataset

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Data_Process():
    
    def __init__(self):

        ###need to change folder every time, directory where photos and csv is
        self.data_dir = '/home/lilaryan/catkin_ws/src/final_test_data'
        
        self.velocities = []
        self.images = []

        #Not sure which image type we can use, whether PIL works or not
        self.load_images()
    
    def get_mean_and_std_of_images(self, data_dir, folder):
        images_mean = np.zeros(3)
        images_std = np.zeros(3)
        count = 0

        files = listdir(data_dir+'/'+folder)
        for file in files:
            file_path = data_dir+'/'+folder+'/'+file
            if file.startswith('image'):
                input_image = Image.open(file_path)
                
                input_array = np.array(input_image) / 255.0  
                
                images_mean += np.mean(input_array, axis=(0, 1))
                images_std += np.std(input_array, axis=(0, 1))
                count += 1

        images_mean = images_mean / count
        images_std = images_std / count

        return images_mean, images_std

        

    def load_images(self):
        """
        Each image is loaded with os library and PIL library and then processed
        following the basic standards specified by this resnet guide:
        
        https://pytorch.org/hub/pytorch_vision_resnet/
        """

        data_dir = self.data_dir

        #see os walk documentation if more info needed
        top_dir = listdir(data_dir)
        for folder in top_dir:
            if folder != ".DS_Store" and folder == 'run_1':
                files = listdir(data_dir+'/'+folder)
                image_mean, image_std = self.get_mean_and_std_of_images(data_dir, folder)

                for file in files:
                    file_path = data_dir+'/'+folder+'/'+file
                    if file.startswith('image'):

                        ###Uses PIL Image, verify if PIL format is usable for us
                        input_image = Image.open(file_path)

                        preprocess = transforms.Compose([
                            transforms.Resize(256),     #change values
                            transforms.CenterCrop(224), #change values
                            transforms.ToTensor(),
                            transforms.Normalize(mean=image_mean, std=image_std),
                        ])

                        input_tensor = preprocess(input_image)
                        
                        ###Unsure if this is needed (or how much image processesing is required)
                        #input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

                        self.images.append(input_tensor)

                    elif file.endswith('.csv'):
                        df = pd.read_csv(file_path, names=['Timestamp', 'Linear Velocity', 'Angular Velocity', 'Quality'], sep=' ')
                        for index, row in df.iterrows():
                            self.velocities.append((row['Linear Velocity'], row['Angular Velocity']))

                while len(self.velocities) != len(self.images):
                    self.velocities.append(self.velocities[len(self.velocities) - 1])        

    def apply_color_jitter(self):
        
        apply_jitter = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ])
        
        # Apply the transformation to each image
        jittered_images = [apply_jitter(image) for image in self.images]
        
        return jittered_images

    def get_image(self):
        self.apply_color_jitter()
        return self.images

    def un_normalize_linear_vel(self, val):
        return ((val * (self.max_lin_val - self.min_lin_vel)) + self.min_lin_vel)
    
    def un_normalize_angular_vel(self,val):
        return ((val * (self.max_ang_val - self.min_ang_vel)) + self.min_ang_vel)   
        

    def get_actions(self):
        data = self.velocities

        lin_vel = [item[0] for item in data]
        ang_vel = [item[1] for item in data]

        # normalize using min-max scaling
        normalized_lin = [(x - min(lin_vel)) / (max(lin_vel) - min(lin_vel)) for x in lin_vel]
        # print(f"normalized linear: {normalized_lin}")
        normalized_ang = [(x - min(ang_vel)) / (max(ang_vel) - min(ang_vel)) for x in ang_vel]
        # print(f"normalized angular: {normalized_ang}")

        self.min_lin_vel = min(lin_vel)
        self.max_lin_val = max(lin_vel)
        self.min_ang_vel = min(ang_vel)
        self.max_ang_val = max(ang_vel)


        normalized_velocities = list(zip(normalized_lin, normalized_ang))

        print(normalized_velocities)

        return self.velocities

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("classification_10_epochs.pth"))
model.eval()

observed_data = Data_Process()

'''
test_set = [(1,(1,1)), (2,(2,2)), (3,(3,3)), (4,(4,4)), (5,(5,5)), (6,(6,6)), (7,(7,7)), (8,(8,8))]
train_size = int(.8 * len(test_set))
test_size = len(test_set) - train_size
train_dataset, test_dataset = random_split(test_set, [train_size, test_size])
print(train_dataset)
'''

image_data = observed_data.get_image()
actions = observed_data.get_actions()
dataset = merge_data(image_data, actions)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

with torch.no_grad():
    for id, data in enumerate(train_loader):
        #only two variables since we currently only have two input states (image, motion), could modify to have more
        #should split up into specified batches of specific size
        #uses model to predict the action, should output an array of predictions
        image, action = data
        pred_action = model(image)

        target_action = torch.stack((action[0], action[1]), dim=1).float()
        #Here we use a mean square error to determine how close/far the predictions are from the actual values
        print(pred_action, target_action)


            
