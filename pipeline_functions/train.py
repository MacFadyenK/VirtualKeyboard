import snntorch as snn
from snntorch import functional as SF

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, TensorDataset, WeightedRandomSampler

def train(model, num_epochs, train_loader, val_loader, criterion, optimizer, device, update_every=5, batch_first=False):
    """
    trains an SNN model for the specified number of epochs
    
    Inputs:
    - model: the SNN model to be trained
    - num_epochs: number of epochs to train for
    - train_loader: pytorch dataloader for the training dataset. Data is the form of (time steps x batch x feature dimension) or
                    (batch x time steps x feature dimension)
    - val_loader: pytorch dataloader for the validation dataset
    - criterion: the loss function to be used to calculate loss Must be from snnTorch and use output
                 spikes, not membrane voltage
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
        # train model for one epoch and append the loss and accuracy to the history
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, batch_first=batch_first)
        training_history["train_loss"].append(train_loss)
        training_history["train_acc"].append(train_acc)

        # check loss and accuracy on validation set and add to the history
        val_loss, val_acc, avg_spk, max_mem, total_counts = validate_snn(model, val_loader, criterion, device, batch_first=batch_first)
        training_history["val_loss"].append(val_loss)
        training_history["val_acc"].append(val_acc)

        # print training status update after the specified number of epochs pass
        if e % update_every == 0:
            print(f"Epoch {e+1}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc*100:.2f}%, " + 
                  f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%")
            print(f"Avg output spike rate: {avg_spk:.4f}")
            print(f"Avg output membrane max: {max_mem:.4f}")
            print(f"Predictions for each class: {total_counts}")
    
    return training_history
        

def train_epoch(model, train_loader, criterion, optimizer, device, batch_first = False):
    """
    trains a SNN model for one epoch
    
    Inputs:
    - model: the SNN model to be trained
    - train_loader: pytorch dataloader for the training dataset
    - criterion: the loss function to be used to calculate loss. Must be from snnTorch and use output
                 spikes, not membrane voltage
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

    model.train()
    # loop through entire training set
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # forward pass
        spk_rec, mem_rec = model(x, batch_first=batch_first)

        # loss calculation
        mem_mean = mem_rec.mean(dim=0)
        loss = criterion(mem_mean, y)

        # calculating gradients and weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # adding batch loss to total loss
        total_loss += loss.item() * spk_rec.size(1)

        # adding batch correct to total correct (assuming rate encoding)
        num_correct += SF.accuracy_rate(spk_rec, y) * spk_rec.size(1)

        #adding to total number in training set
        total += spk_rec.size(1)

    # calculating the average loss and accuracy across the training set
    avg_loss = total_loss/total
    acc = num_correct/total
    return avg_loss, acc

def validate_snn(model, val_loader, criterion, device, batch_first=False):
    """
    evaluates a SNN model on the entire validation set
    
    Inputs:
    - model: the SNN model
    - val_loader: pytorch dataloader for the validation dataset
    - criterion: the loss function to be used to calculate loss. Must be from snnTorch and use output
                 spikes, not membrane voltage
    - device: the device which the model is in. e.g. cuda, cpu
    - batch_first: whether the data has the batch as first dimension or time steps as first dimension

    Returns:
    - avg_loss: total loss across the validation set divided by the number of samples in the val_loader
    - acc: accuracy of the model on the data in val_loader
    - avg_spike_rate: average rate of spiking of the output neurons for the validation set
    - avg_membrane_max: cross batch average of the maximum membrane potential produced by the output LIF neurons
    - total_counts: total number of predictions of the validation set for each class. [nonP300, P300]
    """
    model.eval()
    total_loss = 0.0
    num_correct = 0
    total = 0

    # for spike output monitoring
    total_output_spikes = 0
    total_mem2_max = 0
    total_steps = 0

    # to count classification of each class
    total_counts = torch.zeros(2, dtype=torch.long)

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            # Reset SNN state if model supports it
            # if hasattr(model, "reset"):
            #     model.reset()

            # forward pass
            spk_rec, mem_rec = model(x, batch_first=batch_first)

            # Output spikes
            out_spk_rate = spk_rec.mean().item()
            total_output_spikes += out_spk_rate
            total_mem2_max += mem_rec.max().item()
            total_steps += 1

            # Compute loss on spike trains
            mem_mean = mem_rec.mean(dim=0)
            loss = criterion(mem_mean, y)

            # adding batch loss to total loss
            total_loss += loss.item() * spk_rec.size(1)

            # adding batch correct to total correct (assuming rate encoding)
            num_correct += SF.accuracy_rate(spk_rec, y) * spk_rec.size(1)
            
            spike_counts = spk_rec.sum(dim=0)
            pred = spike_counts.argmax(dim=1)
            class_counts = torch.bincount(pred, minlength=2)
            total_counts += class_counts

            # adding to total number in training set
            total += spk_rec.size(1)

    avg_spk_rate = total_output_spikes/total_steps
    avg_membrane_max = total_mem2_max/total_steps

    avg_loss = total_loss / total
    acc = num_correct / total
    return avg_loss, acc, avg_spk_rate, avg_membrane_max, total_counts

def prepare_training_data(X, y, batch_size, balanced = True):
    """
    transforms numpy arrays for features and labels into dataloaders split for training, validation, and testing

    Inputs:
    - X: numpy feature array 
    - y: numpy data label array
    - batch_size: sample batch size for dataloaders
    - balanced: whether the training data should be balanced between classes or not. Applies a WeightedRandomSampler 
                on the training set. Defaults True for a balanced dataset

    Outputs:
    - train_loader: pytorch dataloader for training dataset
    - val_loader: pytorch dataloader for validation dataset
    - test_loader: pytorch dataloader for testing dataset
    """
    # change into pytorch tensors
    X_tensor = torch.from_numpy(X)
    X_tensor = X_tensor.clone().detach().float()
    y_tensor = torch.from_numpy(y)

    # load into a tensor dataset
    full_dataset = TensorDataset(X_tensor, y_tensor)

    # separate dataset into training, validation, and testing
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    lengths = [train_size, val_size, test_size]
    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, lengths)

    # creating a sampler out P300 and non-P300 samples for training set so they are closer to 50/50
    train_labels = torch.tensor([full_dataset[i][1] for i in train_dataset.indices])
    class_counts = torch.bincount(train_labels)
    print("Training Class Counts: ", class_counts)
    # Calculate inverse frequencies
    class_weights = 1.0 / class_counts

    # Normalize weights (optional, but often helpful)
    class_weights = class_weights / class_weights.sum() * len(class_counts) 

    print("Training Class Weights:", class_weights)
    
    train_loader=None

    if balanced:
        # Assign weight to each sample based on its class
        sample_weights = class_weights[train_labels]

        sampler = WeightedRandomSampler(
            sample_weights, 
            len(sample_weights), 
            replacement=True # Allows oversampling of minority classes
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, class_weights