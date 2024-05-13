#!/usr/bin/env python3

import numpy as np
import torch as th
from torchvision import transforms
from torchvision.transforms import v2
from os import walk
import pandas as pd

from PIL import Image



class Data_Process():
    
    def __init__(self):

        ###need to change folder every time, directory where photos and csv is
        self.data_dir = '/home/gaversa/catkin_ws/src/final_project_soccer_bot/data/data_coll_5_9'

        #Not sure which image type we can use, whether PIL works or not
        self.load_images()

        self.expert_dset = None
        self.runs = []
        #self.images = []
        #self.velocities = []

    def load_images(self):
        """
        Each image is loaded with os library and PIL library and then processed
        following the basic standards specified by this resnet guide:
        
        https://pytorch.org/hub/pytorch_vision_resnet/
        """
        cur_run = 0
        images = []
        velocities = []

        #see os walk documentation if more info needed
        for (top_dirpath, top_dirnames, top_filenames) in walk(photo_dir):
            for (dirpath, dirnames, filenames) in walk(top_dirnames):
                if filenames.startswith('image'):
                    ###Uses PIL Image, verify if PIL format is usable for us
                    input_image = Image.open(filenames)

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
                    velocities.append(actions)

                elif filenames.endswith('.csv'):
                    df = pd.read_csv(filenames)
                    for index, row in df.iterrows():
                        velocities.append(row['Linear Velocity'], row['Angular Velocity'])
                        
            
            ExpertDataSet(cur_run, images, velocities)
            cur_run += 1
            images = []
            velocities = []
            





class ExpertDataSet(Dataset):
   
   def __init__(self, index, image_list, csv_list):
        """
        Data set containing the data for a single run.

        Index stores what number run
        Observations stores all images as a list in chronological order
        Actions stores all velocities read from the csv as a tuple pair 
        (Linear, Angular) in order of each time stamp
        """
        self.index = index
        self.observations = image_list
        self.actions = csv_list

   def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

   def __len__(self):
        return len(self.observations)