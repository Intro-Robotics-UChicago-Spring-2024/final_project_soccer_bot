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

#runs the training and testing for the model
def main():
    #loading in the dataset that we collected
    observed_data = Data_Process()
    #load datasets, should be represented as (state, action) pair
    #observations should be images, actions should be linear and angular velocity
    image_data = observed_data.get_image()
    actions = observed_data.get_actions()
    #makes it a dataset to feed into the dataloader
    dataset = SoccerBotDataset(image_data, actions)
    #gets the size of the list so we know how many classes there are
    observed_data_lin_size, observed_data_ang_size = observed_data.get_ang_lin()

    #initializes the model
    model = MyModel(observed_data_lin_size + 1, observed_data_ang_size + 1)

    #Use cross entropy loss for classification
    loss_function = nn.CrossEntropyLoss()

    #need to change and look which one is best for us
    # training parameters
    lr = 1.0e-2
    momentum = 0.9
    weight_decay = 1.0e-4

    #optimizer for model, used for gradients
    optimizer = torch.optim.SGD(model.parameters(), lr,
                                momentum=momentum,
                                weight_decay=weight_decay)

    #determines the sizes of the training and testing datasets
    train_size = int(.8 * len(dataset))
    test_size = len(dataset) - train_size

    #must compute sizes and not just input fractions to avoid rounding errors and maintain consistency
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    #used to be able to load the data properly into model, allows to split a batchsize and returns tensorflow array
    #I have a quad core processor which is why I put num_workers as 4 but we can vary depending on the hardware we run this on
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=4)

    #For loop runs the test and train and creates a model
    num_epochs = 10
    for epoch in range(0, num_epochs):
        #trains model and outputs the average loss
        loss_from_train = train(model, train_loader, optimizer, loss_function)
        print("loss from train:" + str(loss_from_train))
        #test/validates model and outputs the average loss
        prediction = test(model, test_loader, optimizer, loss_function)
        print(prediction)
    
    #saves created model in the same repo
    torch.save(model.state_dict(), 'soccer_bot_classification_18.pth')

#dataset specifically made for our model. Used because dataloader has to take in data in the form of a pytorch dataset
class SoccerBotDataset(Dataset):
    def __init__(self, image, actions, transform=None):
        self.image = image
        self.actions = actions

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        image = self.image[idx]
        action = self.actions[idx]
        return image, torch.tensor(action, dtype=torch.float32)

#Initializes the model
class MyModel(nn.Module):
    def __init__(self, num_classes1, num_classes2):
        super(MyModel, self).__init__()
        #using a resnet 18 but can make it a different one
        self.model_resnet = models.resnet18(pretrained=True)
        num_ftrs = self.model_resnet.fc.in_features
        self.model_resnet.fc = nn.Identity()
        #Adding two layers, one for linear velo and one for angular velo
        self.fc1 = nn.Linear(num_ftrs, num_classes1)
        self.fc2 = nn.Linear(num_ftrs, num_classes2)

    #Out put of the model. Since were using classification and two layers we want two individual outputs
    def forward(self, x):
        x = self.model_resnet(x)
        out1 = self.fc1(x)
        out2 = self.fc2(x)
        return out1, out2

#Training function for our model
def train(model, train_loader, optimizer, loss_function):
    #will tell us the average loss
    losses = AverageMeter()
    #setting model to train mode
    model.train()

    #data loader should contain batches of current images and current 
    for id, data in enumerate(train_loader):
        #only two variables since we currently only have two input states (image, motion), could modify to have more
        #should split up into specified batches of specific size
        image, action = data
        lin_v = action[:, 0].long()
        ang_v = action[:, 1].long()
        #zeros out the gradient for batch
        optimizer.zero_grad()
        #uses model to predict the action, should output an array of predictions
        pred_action_lin, pred_action_ang = model(image)

        #Here we use classification to determine how close/far the predictions are from the actual values
        loss_lin = loss_function(pred_action_lin, lin_v)
        loss_ang = loss_function(pred_action_ang, ang_v)

        loss = loss_lin + loss_ang
        #computes gradients
        loss.backward()
        #updates the parameters of the model using the gradients, (parameters are the learnable components)
        optimizer.step()

        # measure accuracy and record loss
        with torch.no_grad():     
            losses.update(loss.detach().cpu().item(), image.size(0))

    return losses.avg

#Testing function for out model
def test(model, test_loader, loss_function):
    #records average loss
    losses_test = AverageMeter()
    #sets model into evaluation mode so we can test
    model.eval()
    #Makes sure model isn't changed/gradients aren't updated
    with torch.no_grad():
        #data loader should contain batches of current images and current 
        for id, data in enumerate(test_loader):
            #only two variables since we currently only have two input states (image, motion), could modify to have more
            #should split up into specified batches of specific size
            #uses model to predict the action, should output an array of predictions
            image, action = data
            lin_v = action[:, 0].long()
            ang_v = action[:, 1].long()
            pred_action_lin, pred_action_ang = model(image)

            #Here we use classification to determine how close/far the predictions are from the actual values
            loss_lin = loss_function(pred_action_lin, lin_v)
            loss_ang = loss_function(pred_action_ang, ang_v)

            loss = loss_lin + loss_ang

            losses_test.update(loss.detach().cpu().item(), image.size(0))

    return losses_test.avg

#class that creates the meter to monitor our average loss, can be extended to other values/variables
class AverageMeter(object):
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

#This is where we pull our data and transform it so we can use it in our model
class Data_Process():
    
    def __init__(self):

        ###need to change folder every time, directory where photos and csv is
        #Path to data dir
        self.data_dir = '/home/gnakanishi/catkin_ws/src/final_project_soccer_bot/test_data'
        
        #storage variables
        self.velocities = []
        self.lin_vel = []
        self.ang_vel = []
        self.images = []
        self.min_lin_vel = 0
        self.min_ang_vel = 0

        self.load_images()

    #Loads images from data folder and transforms them to use in model
    def load_images(self):
        """
        Each image is loaded with os library and PIL library and then processed
        following the basic standards specified by this resnet guide:
        
        https://pytorch.org/hub/pytorch_vision_resnet/
        """

        data_dir = self.data_dir

        #see os walk documentation if more info needed
        #This loop walks through the directories and reads and opens all of the image files.
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

    #returns the image array
    def get_image(self):
        return self.images
    
    #returns the minimum lin and ang velociteis
    def get_min(self):
        return self.min_lin_vel, self.min_ang_vel

    #gets the ang and lin velo while also binning the data
    def get_actions(self):
        data = self.velocities

        #Bins the data
        self.lin_vel = [round(item[0] * 100) for item in data]
        self.ang_vel = [round(item[1] * 100) for item in data]
        self.min_lin_vel = min(self.lin_vel)
        self.min_ang_vel = min(self.ang_vel)

        self.lin_vel = [num + abs(self.min_lin_vel) for num in self.lin_vel]
        self.ang_vel = [num + abs(self.min_ang_vel) for num in self.ang_vel]

        #makes the data into a tuple to be used as a state action pair, this is the action aspect
        normalized_velocities = list(zip(self.lin_vel, self.ang_vel))

        return normalized_velocities

    #returns the range of the ang and lin velo
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
