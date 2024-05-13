#!/usr/bin/env python3

import numpy as np
import torch as th
from torchvision import transforms
from torchvision.transforms import v2
from os import walk
from os import listdir
import pandas as pd

from PIL import Image



class Data_Process():
    
    def __init__(self):

        ###need to change folder every time, directory where photos and csv is
        self.data_dir = '/home/gaversa/catkin_ws/src/final_project_soccer_bot/data/data_coll_5_9'

        #Not sure which image type we can use, whether PIL works or not
        self.load_images()

        #self.expert_dset = None
        self.runs = []

    def load_images(self):
        """
        Each image is loaded with os library and PIL library and then processed
        following the basic standards specified by this resnet guide:
        
        https://pytorch.org/hub/pytorch_vision_resnet/
        """
        cur_run = 0
        images = []
        velocities = []

        data_dir = self.data_dir

        #see os walk documentation if more info needed
        top_dir = listdir(data_dir)
        for folder in top_dir:
            files = listdir(folder)
            for file in files():
    
                if file.startswith('image'):
                    ###Uses PIL Image, verify if PIL format is usable for us
                    input_image = Image.open(file)

                    preprocess = transforms.Compose([
                        transforms.Resize(256),     #change values
                        transforms.CenterCrop(224), #change values
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ])

                    input_tensor = preprocess(input_image)
                    
                    ###Unsure if this is needed (or how much image processesing is required)
                    #input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

                    images.append(input_tensor)

                elif file.endswith('.csv'):
                    df = pd.read_csv(file, names=['Timestamp', 'Linear Velocity', 'Angular Velocity', 'Quality'], sep=' ')
                    for index, row in df.iterrows():
                        velocities.append((row['Timestamp'], row['Linear Velocity'], row['Angular Velocity'], row['Quality']))
                    
            
                run = ExpertDataSet(cur_run, images, velocities)
                self.runs.append(run)
                cur_run += 1
                images = []
                velocities = []
            





class ExpertDataSet(Dataset):
   
   def __init__(self, index, image_list, csv_list):
        """
        Data set containing the data for a single run.

        Index stores what number run
        Observations stores all images as a list in chronological order
        Actions stores all velocities read from the csv as a tuple 
        (Timestamp, Linear, Angular, Quality) in order of each time stamp, where
        quality denotes either 'Good' or 'Bad' runs (where the goal is not
        reached).
        """
        self.index = index
        self.observations = image_list
        self.actions = csv_list

   def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

   def __len__(self):
        return len(self.observations)