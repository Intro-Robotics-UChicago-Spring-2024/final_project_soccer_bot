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
        self.data_dir = '/home/gaversa/catkin_ws/src/final_project_soccer_bot'

        #Not sure which image type we can use, whether PIL works or not
        self.load_images()

        self.expert_dset = None

    def load_images(self):
        """
        Each image is loaded with os library and PIL library and then processed
        following the basic standards specified by this resnet guide:
        
        https://pytorch.org/hub/pytorch_vision_resnet/
        """


        ###rename to fit our naming convention
        photo_dir = self.data_dir + '/photos'
        csv_dir = self.data_dir + '/data_coll_5_8'

        df = pd.read_csv(csv_dir)
        cur_idx = 0

        #see os walk documentation if more info needed
        for (dirpath, dirnames, filenames) in walk(photo_dir):
            
            ###Uses PIL Image, verify if PIL format is usable for us
            input_image = Image.open(filenames)

            preprocess = transforms.Compose([
                transforms.Resize(256),     #change values
                transforms.CenterCrop(224), #change values
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

            ### Assumes df rows are loaded with indexing lining up to the order that photos are iterated over
            actions = (df['Linear Velocity'].iloc[cur_idx], df['Angular Velocity'].iloc[cur_idx])
            self.expert_dset = ExpertDataSet(input_batch, actions)

            cur_idx += 1






class ExpertDataSet(Dataset):
   
   def __init__(self, expert_observations, expert_actions):
        """
        Expert Data Set where self.observations is a [list? batch? of multiple?
        /or single?] image(s) and self.actions is a tuple of the linear and 
        angular velocity for the specified image at a given location
        """
        self.observations = expert_observations
        self.actions = expert_actions

   def __getitem__(self, index):
        return (self.observations[index], self.actions[index])

   def __len__(self):
        return len(self.observations)