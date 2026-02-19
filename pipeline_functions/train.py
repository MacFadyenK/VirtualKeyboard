import snntorch as snn
from snntorch import functional as SF

import torch
import torch.nn as nn

def train(model, num_epochs, train_loader, val_loader, criterion, optimizer, device, update_every=5, batch_first=False):
    """
    trains an SNN model for the specified number of epochs
    
    Inputs:
    - model: the SNN model to be trained
    - num_epochs: number of epochs to train for
    - train_loader: pytorch dataloader for the training dataset. Data is the form of (time steps x batch x feature dimension) or
                    (batch x time steps x feature dimension)
    - val_loader: pytorch dataloader for the validation dataset
    - criterion: the loss function to be used to calculate loss
    - optimizer: the optimizer model to be used for training
    - device: the device which the model is in. e.g. cuda, cpu
    - update_every: Positive integer. Prints training loss, training accuracy, validation loss, and validation accuracy for epochs
                    divisible by update_every. If no number given, prints every 5 epochs
    - batch_first: whether the data has the batch as first dimension or time steps as first dimension

    Returns:
    - training_history: a dictionary with 4 key-value pairs , 
                        key - train_loss: value - list of the average loss from the train_loader dataset set for each epoch
                        key - val_loss: value - list of the average loss from the val_loader dataset for each epoch
                        key - train_acc: value - list of the accuracy from the train_loader dataset for each epoch
                        key - val_acc: value - list of the accuracy from the val_loader dataset for each epoch
    """
    training_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    # loop through all epochs
    for e in range(num_epochs):
        # train model for one epoch and append the loss to the loss history
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, batch_first=batch_first)
        training_history["train_loss"].append(train_loss)
        training_history["train_acc"].append(train_acc)

        ## MARYS VALIDATION FUNCTION GOES HERE TO GET VALIDATION LOSS AND ACCURACY ##

        # print training status update after the specified number of epochs pass
        if e % update_every == 0:
            print(f"Epoch {e}: Training Loss: {train_loss}, Training Accuracy: {train_acc}, ") #ADD VALIDATION LOSS + ACC PRINTING HERE
    
    return training_history
        

def train_epoch(model, train_loader, criterion, optimizer, device, batch_first = False):
    """
    trains a SNN model for one epoch
    
    Inputs:
    - model: the SNN model to be trained
    - train_loader: pytorch dataloader for the training dataset
    - criterion: the loss function to be used to calculate loss
    - optimizer: the optimizer model to be used for training
    - device: the device which the model is in. e.g. cuda, cpu
    - batch_first: whether the data has the batch as first dimension or time steps as first dimension

    Returns:
    - avg_loss: total loss across the entire epoch divided by the number of samples in the train_loader
    - acc: accuracy of the model on the data in train_loader through training for one epoch
    """
    total_loss = 0
    avg_loss = -1
    num_correct = 0
    total = 0
    acc = -1
    # loop through entire training set
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # forward pass
        model.train()
        spk_rec, _ = model(x, batch_first=batch_first)

        # loss calculation
        loss = torch.zeros((1), device=device)
        for step in range(spk_rec.size(0)):
            loss += criterion(spk_rec[step], y)

        # calculating gradients and weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # adding batch loss to total loss
        total_loss += loss.item()

        # adding batch correct to total correct (assuming rate encoding)
        num_correct += SF.accuracy_rate(spk_rec, y) * spk_rec.size(1)

        #adding to total number in training set
        total += spk_rec.size(1)

    # calculating the average loss and accuracy across the training set
    avg_loss = total_loss/total
    acc = num_correct/total
    return avg_loss, acc