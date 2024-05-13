
import torch
import torch.nn as nn
import resnet
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset, random_split

def main():
    #load test data and split into train and validation, in this instance it would be the expert runs
    #make sure to process the data so its ready for consumtion

    #here is where we would load the model and loss function
    #model = resent model
    #loss_function = loss function that I would have to create

    #next we would train the model, training itself will be writen in another function
    #must specify epochs

    #make sure resnet14 is an option
    model = resnet.resnet14(pretrained=False, num_classes=5, input_channels=1)

    loss_function = nn.MSELoss()

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
    
    #load datasets, should be represented as (state, action) pair
    #observations should be images, actions should be linear and angular velocity
    #need to figure out how to combine
    obsevations = 0
    actions = 0
    dataset = 0

    train_size = int(.8 * len(dataset))
    test_size = len(dataset) - train_size

    #must compute sizes and not just input fractions to avoid rounding errors and maintain consistency
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    #I have a quad core processor which is why I put num_workers as 4 but we can vary depending on the hardware we run this on
    train_loader = DataLoader(train_dataset, batchsize=128, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batchsize=128, shuffle=True, num_workers=4)

    num_epochs = 10
    for epoch in range(0, num_epochs):
        loss_from_train = train(model, train_loader, optimizer, loss_function)
        prediction = test(model, test_loader, optimizer, loss_function)
    
    torch.save(model.state_dict(), 'soccer_bot_model.pth')






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
        pred_action = model(image)

        #Here we use a mean square error to determine how close/far the predictions are from the actual values
        loss = loss_function(pred_action, action)
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
            image, action = data
            #uses model to predict the action, should output an array of predictions
            pred_action = model(image)
            #Here we use a mean square error to determine how close/far the predictions are from the actual values
            loss = loss_function(pred_action, action)
            losses_test.update(loss.detach().cpu().item(), image.size(0))

            print(loss)

    return losses_test.avg


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



if __name__ == '__main__':
    main()
