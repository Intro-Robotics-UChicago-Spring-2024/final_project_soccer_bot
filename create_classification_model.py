#!/usr/bin/env python3

import torch
import torchvision.models as models
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, random_split
from torchvision import transforms
from torchvision.transforms import v2
from os import walk
from os import listdir
import pandas as pd
import numpy as np
import math

from PIL import Image

def main():
    #load test data and split into train and validation, in this instance it would be the expert runs
    #make sure to process the data so its ready for consumtion

    #here is where we would load the model and loss function
    #model = resent model
    #loss_function = loss function that I would have to create

    #next we would train the model, training itself will be writen in another function
    #must specify epochs

    observed_data = Data_Process()
    #load datasets, should be represented as (state, action) pair
    #observations should be images, actions should be linear and angular velocity
    #need to figure out how to combine
    image_data = observed_data.get_image()
    actions = observed_data.get_actions()
    dataset = merge_data(image_data, actions)
    observed_data_lin_size, observed_data_ang_size = observed_data.get_ang_lin()

    model = MyModel(observed_data_lin_size + 2, observed_data_ang_size + 2)

    loss_function = nn.CrossEntropyLoss()

    #need to change and look which one is best for us
    # training parameters
    lr = 1.0e-2
    momentum = 0.9
    weight_decay = 1.0e-4
    batchsize = 64
    batchsize_valid = 64
    start_epoch = 0
    epochs      = 50000
    nbatches_per_epoch = int(epochs/batchsize)
    nbatches_per_valid = int(epochs/batchsize_valid)

    #need to change and figure out which one is best for us
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    train_size = int(.8 * len(dataset))
    test_size = len(dataset) - train_size

    #must compute sizes and not just input fractions to avoid rounding errors and maintain consistency
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    #I have a quad core processor which is why I put num_workers as 4 but we can vary depending on the hardware we run this on
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)

    num_epochs = 10
    for epoch in range(start_epoch, num_epochs):
        loss_from_train = train(model, train_loader, optimizer, loss_function)
        print("loss from train:" + str(loss_from_train))
        prediction = test(model, test_loader, optimizer, loss_function)
        print(prediction)
    
    torch.save(model.state_dict(), 'soccer_bot_classification_18.pth')

class MyModel(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super(MyModel, self).__init__()
        self.model_resnet = models.resnet18(pretrained=True)
        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        self.fc1 = nn.Linear(num_ftrs, num_classes1)
        self.fc2 = nn.Linear(num_ftrs, num_classes2)

    def forward(self, x):
        x = self.model_resnet(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2

def train(model, train_loader, optimizer, loss_function):
    #will tell us the average loss
    losses = AverageMeter()
    
    model.train()

    #data loader should contain batches of current images and current 
    for id, data in enumerate(train_loader):
        #only two variables since we currently only have two input states (image, motion), could modify to have more
        #should split up into specified batches of specific size
        image, action = data
       
        #zeros out the gradient for batch
        optimizer.zero_grad()
        #uses model to predict the action, should output an array of predictions
        pred_action_lin, pred_action_ang = model(image)

        #target_action = torch.stack((action[0], action[1]), dim=1).float()

        #Here we use a mean square error to determine how close/far the predictions are from the actual values
        loss_lin = loss_function(pred_action_lin, action[0])
        loss_ang = loss_function(pred_action_ang, action[1])

        loss = loss_lin + loss_ang
        #computes gradients
        loss.backward()
        #updates the parameters of the model using the gradients, (parameters are the learnable components)
        optimizer.step()

        # measure accuracy and record loss
        with torch.no_grad():     
            losses.update(loss.detach().cpu().item(), image.size(0))

    return losses.avg


def test(model, test_loader, optimizer, loss_function):
    losses_test = AverageMeter()
    model.eval()

    with torch.no_grad():
        #data loader should contain batches of current images and current 
        for id, data in enumerate(test_loader):
            #only two variables since we currently only have two input states (image, motion), could modify to have more
            #should split up into specified batches of specific size
            #uses model to predict the action, should output an array of predictions
            image, action = data
            pred_action_lin, pred_action_ang = model(image)

            #target_action = torch.stack((action[0], action[1]), dim=1).float()

            #Here we use a mean square error to determine how close/far the predictions are from the actual values
            loss_lin = loss_function(pred_action_lin, action[0])
            loss_ang = loss_function(pred_action_ang, action[1])

            loss = loss_lin + loss_ang

            losses_test.update(loss.detach().cpu().item(), image.size(0))

    return losses_test.avg

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
        self.data_dir = '/home/gnakanishi/catkin_ws/src/final_project_soccer_bot/test_data'
        
        self.velocities = []
        self.lin_vel = []
        self.ang_vel = []
        self.images = []

        #Not sure which image type we can use, whether PIL works or not
        self.load_images()

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
            if folder != ".DS_Store":
                files = listdir(data_dir+'/'+folder)
                for file in files:
                    file_path = data_dir+'/'+folder+'/'+file
                    if file.startswith('image'):

                        ###Uses PIL Image, verify if PIL format is usable for us
                        input_image = Image.open(file_path)

                        preprocess = transforms.Compose([
                            transforms.Resize(256),     #change values
                            transforms.CenterCrop(224), #change values
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

    def get_image(self):
        return self.images

    def get_actions(self):
        data = self.velocities

        self.lin_vel = [round(item[0] * 100) for item in data]
        self.ang_vel = [round(item[1] * 100) for item in data]

        '''
        min_lin_velo = math.ceil(min(lin_vel))
        max_lin_velo = math.ceil(max(lin_vel))
        num_bins_lin_velo = max_lin_velo - min_lin_velo

        bins_lin_velo = np.linspace(min_lin_velo, max_lin_velo, num_bins_lin_velo + 1)  
        binned_data_lin_velo = [[] for _ in range(num_bins_lin_velo + 1)]  

        for value in lin_vel:
            if value == 0:  # Handle zero as a special case
                bin_index = num_bins_lin_velo  # Set bin index to the last bin (zero bin)
            else:
                bin_index = np.digitize(value, bins_lin_velo) - 1  # Find the index of the bin for the value
            if bin_index >= 0 and bin_index <= num_bins_lin_velo:  # Check if the bin index is valid
                binned_data_lin_velo[bin_index].append(value) 


        min_ang_velo = math.ceil(min(ang_vel))
        max_ang_velo = math.ceil(max(ang_vel))
        num_bins_ang_velo = max_ang_velo - min_ang_velo

        bins_ang_velo = np.linspace(min_ang_velo, max_ang_velo, num_bins_ang_velo + 1)  
        binned_data_ang_velo = [[] for _ in range(bins_ang_velo + 1)]  

        for value in ang_vel:
            if value == 0:  # Handle zero as a special case
                bin_index = num_bins_ang_velo  # Set bin index to the last bin (zero bin)
            else:
                bin_index = np.digitize(value, bins_ang_velo) - 1  # Find the index of the bin for the value
            if bin_index >= 0 and bin_index <= num_bins_ang_velo:  # Check if the bin index is valid
                binned_data_ang_velo[bin_index].append(value) 

        # normalize using min-max scaling
        normalized_lin = [(x - min(lin_vel)) / (max(lin_vel) - min(lin_vel)) for x in lin_vel]
        # print(f"normalized linear: {normalized_lin}")
        normalized_ang = [(x - min(ang_vel)) / (max(ang_vel) - min(ang_vel)) for x in ang_vel]
        # print(f"normalized angular: {normalized_ang}")
        '''

        normalized_velocities = list(zip(self.lin_vel, self.ang_vel))

        return normalized_velocities

    def get_ang_lin(self):
        lin_vel_min = min(self.lin_vel)
        lin_vel_max = max(self.lin_vel)
        lin_vel_range = lin_vel_max - lin_vel_min

        ang_vel_min = min(self.ang_vel)
        ang_vel_max = max(self.ang_vel)
        ang_vel_range = ang_vel_max - ang_vel_min
        
        return lin_vel_range, ang_vel_range



if __name__ == '__main__':
    main()
