import snntorch as snn
from snntorch import functional as SF

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset, WeightedRandomSampler

def train(model, num_epochs, train_loader, val_loader, criterion, optimizer, scheduler, device, loss_style = 'spk', update_every=5, checkpoint_path=None, batch_first=False):
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
    - loss_style: whether to compute loss based on spikes or on membrane potential: 
            for spikes, enter 'spk', for membrane potential, enter 'mem'
    - update_every: Positive integer. Prints training loss, training accuracy, validation loss, and validation accuracy for epochs
                    divisible by update_every. If no number given, prints every 5 epochs
    - checkpoint_path: file path to the location which to save the training checkpoints
    - batch_first: whether the data has the batch as first dimension or time steps as first dimension

    Returns:
    - training_history: a dictionary with 4 key-value pairs , 
                        key - train_loss: value - list of the average loss from the train_loader dataset set for each epoch
                        key - val_loss: value - list of the average loss from the val_loader dataset for each epoch
                        key - train_acc: value - list of the accuracy from the train_loader dataset for each epoch
                        key - val_acc: value - list of the accuracy from the val_loader dataset for each epoch
    """
    training_history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "train_bal_acc": [], "val_bal_acc": []}
    # loop through all epochs
    for e in range(num_epochs):
        # train model for one epoch and append the loss and accuracy to the history
        train_loss, train_acc, train_counts, train_bal_acc = train_epoch(model, train_loader, criterion, optimizer, device, loss_style=loss_style, batch_first=batch_first)
        training_history["train_loss"].append(train_loss)
        training_history["train_acc"].append(train_acc)
        training_history["train_bal_acc"].append(train_bal_acc)

        # check loss and accuracy on validation set and add to the history
        val_loss, val_acc, avg_spk, max_mem, total_counts, val_bal_acc = validate_snn(model, val_loader, criterion, device, loss_style=loss_style, batch_first=batch_first)
        training_history["val_loss"].append(val_loss)
        training_history["val_acc"].append(val_acc)
        training_history["val_bal_acc"].append(val_bal_acc)

        scheduler.step(val_loss)

        # print training status update after the specified number of epochs pass
        if (e+1) % update_every == 0:
            
            if checkpoint_path is not None: # saving a checkpoint
                checkpoint = f'{checkpoint_path}/checkpoint_e{e+1}.tar'
                save_checkpoint(e, model=model, optimizer=optimizer, scheduler = scheduler, loss=train_loss, history=training_history, path=checkpoint)
            
            print(f"Epoch {e+1}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc*100:.2f}%, Training Balanced Accuracy: {train_bal_acc*100:.2f}%, \n" + 
                  f"    Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%, Validation Balanced Accuracy: {val_bal_acc*100:.2f}%")
            print(f"    Avg output spike rate: {avg_spk:.4f}")
            print(f"    Avg output membrane max: {max_mem:.4f}")
            print(f"    Predictions for each class on Training Set: {train_counts}")
            print(f"    Predictions for each class on Val Set: {total_counts}")
    
    return training_history
        

def train_epoch(model, train_loader, criterion, optimizer, device,loss_style='spk', batch_first = False):
    """
    trains a SNN model for one epoch
    
    Inputs:
    - model: the SNN model to be trained
    - train_loader: pytorch dataloader for the training dataset
    - criterion: the loss function to be used to calculate loss. Must be from snnTorch and use output
                 spikes, not membrane voltage
    - optimizer: the optimizer model to be used for training
    - device: the device which the model is in. e.g. cuda, cpu
    - loss_style: whether to compute loss based on spikes or on membrane potential: 
                for spikes, enter 'spk', for membrane potential, enter 'mem'
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

    TP, FN, TN, FP = 0, 0, 0, 0

    # to count classification of each class
    total_counts = torch.zeros(2, dtype=torch.long, device=device)

    model.train()
    # loop through entire training set
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        # forward pass
        spk_rec, mem_rec = model(x, batch_first=batch_first)

        # loss calculation
        # Compute loss on spike trains
        if loss_style == 'mem':
            mem_mean = mem_rec.mean(dim=0)
            loss = criterion(mem_mean, y)
            # spike_rate = spk_rec.mean()
            # if spike_rate > 0.15:
            #     loss += 0.1 * (spike_rate - 0.15) ** 2
        elif loss_style == 'spk':
            spike_counts = spk_rec.sum(dim=0)
            spike_rates = spike_counts / spk_rec.size(0)  # per neuron, normalized over time
            loss = criterion(spike_rates, y)
            #spike regularization to encourage lower spiking rate ~0.35
            loss = loss + 0.05 * (spk_rec.mean() - 0.35)**2
        elif loss_style == 'mse':
            loss = criterion(spk_rec, y)
        elif loss_style == 'hybrid':
            mem_mean = mem_rec.mean(dim=0)
            spike_counts = spk_rec.sum(dim=0)
            spike_rates = spike_counts / spk_rec.size(0)
            loss = criterion(spike_rates, y) + 0.5* criterion(mem_mean, y)
        else:
            raise ValueError("Invalid loss type input")

        # calculating gradients and weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # adding batch loss to total loss
        total_loss += loss.item() * y.size(0)
        
        # predictions for each class
        spike_counts = spk_rec.sum(dim=0)
        pred = spike_counts.argmax(dim=1)

        num_correct += (pred == y).sum().item() # adding batch correct to total correct

        # confusion matrix values
        TP += ((pred == 1) & (y == 1)).sum().item()
        FN += ((pred == 0) & (y == 1)).sum().item()
        FP += ((pred == 1) & (y == 0)).sum().item()
        TN += ((pred == 0) & (y == 0)).sum().item()

        # how many of each class was predicted
        class_counts = torch.bincount(pred, minlength=2)
        total_counts += class_counts

        #adding to total number in training set
        total += y.size(0)

    # calculating the average loss and accuracies across the training set
    avg_loss = total_loss/total
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    bal_acc = (sensitivity+specificity)/2
    acc = num_correct/total

    return avg_loss, acc, total_counts, bal_acc

def validate_snn(model, val_loader, criterion, device, loss_style='spk', batch_first=False):
    """
    evaluates a SNN model on the entire validation set
    
    Inputs:
    - model: the SNN model
    - val_loader: pytorch dataloader for the validation dataset
    - criterion: the loss function to be used to calculate loss. Must be from snnTorch and use output
                 spikes, not membrane voltage
    - device: the device which the model is in. e.g. cuda, cpu
    - loss_style: whether to compute loss based on spikes or on membrane potential: 
                  for spikes, enter 'spk', for membrane potential, enter 'mem'
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
    total_counts = torch.zeros(2, dtype=torch.long, device=device)

    TP, FN, TN, FP = 0, 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            # forward pass
            spk_rec, mem_rec = model(x, batch_first=batch_first)

            # Output spikes
            out_spk_rate = spk_rec.mean().item()
            total_output_spikes += out_spk_rate
            total_mem2_max += mem_rec.max().item()
            total_steps += 1

            # Compute loss on spike trains
            if loss_style == 'mem':
                mem_mean = mem_rec.mean(dim=0)
                loss = criterion(mem_mean, y)
                # spike_rate = spk_rec.mean()
                # if spike_rate > 0.15:
                #     loss += 0.1 * (spike_rate - 0.15) ** 2
            elif loss_style == 'spk':
                spike_counts = spk_rec.sum(dim=0)
                spike_rates = spike_counts / spk_rec.size(0)  # per neuron, normalized over time
                loss = criterion(spike_rates, y)
                #spike regularization to encourage lower spiking rate ~0.35
                loss = loss + 0.05 * (spk_rec.mean() - 0.35)**2
            elif loss_style == 'mse':
                loss = criterion(spk_rec, y)
            elif loss_style == 'hybrid':
                mem_mean = mem_rec.mean(dim=0)
                spike_counts = spk_rec.sum(dim=0)
                spike_rates = spike_counts / spk_rec.size(0)
                loss = criterion(spike_rates, y) + 0.5* criterion(mem_mean, y)
            else:
                raise ValueError("Invalid loss type input")

            # adding batch loss to total loss
            total_loss += loss.item() * y.size(0)

            #get predictions
            spike_counts = spk_rec.sum(dim=0)
            pred = spike_counts.argmax(dim=1)

            num_correct += (pred == y).sum().item() # adding batch correct to total correct

            # confusion matrix values
            TP += ((pred == 1) & (y == 1)).sum().item()
            FN += ((pred == 0) & (y == 1)).sum().item()
            FP += ((pred == 1) & (y == 0)).sum().item()
            TN += ((pred == 0) & (y == 0)).sum().item()

            # counting how many of each class is predicted
            class_counts = torch.bincount(pred, minlength=2)
            total_counts += class_counts

            # adding to total number in training set
            total += y.size(0)

    # getting spike rate and membrane max
    avg_spk_rate = total_output_spikes/total_steps
    avg_membrane_max = total_mem2_max/total_steps

    # calculating final evaluation values
    avg_loss = total_loss / total
    sensitivity = TP/(TP+FN + 1e-8)
    specificity = TN/(TN+FP + 1e-8)
    bal_acc = (sensitivity+specificity)/2
    acc = num_correct / total

    return avg_loss, acc, avg_spk_rate, avg_membrane_max, total_counts, bal_acc

def save_checkpoint(epoch, model, optimizer, scheduler, loss, history, path):
    """Saves the training state to a file."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': loss,
        'history': history
    }
    torch.save(state, path)

def prepare_training_data(X, y, batch_size, balanced = True, seed=42):
    """
    transforms numpy arrays for features and labels into dataloaders split for training, validation, and testing

    Inputs:
    - X: numpy feature array 
    - y: numpy data label array
    - batch_size: sample batch size for dataloaders
    - balanced: whether the training data should be balanced between classes or not. Applies a WeightedRandomSampler 
                on the training set. Defaults True for a balanced dataset
    - seed: random seed for reproducibility of dataset splits and dataloader shuffling. Default 42

    Outputs:
    - train_loader: pytorch dataloader for training dataset
    - val_loader: pytorch dataloader for validation dataset
    - test_loader: pytorch dataloader for testing dataset
    """
    # change into pytorch tensors
    X_tensor = torch.from_numpy(X).float()
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

    g = torch.Generator().manual_seed(seed) # for reproducibility of splits
    lengths = [train_size, val_size, test_size]
    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, lengths, generator=g)

    # creating a sampler out P300 and non-P300 samples for training set so they are closer to 50/50
    train_labels = y_tensor[train_dataset.indices]
    class_counts = torch.bincount(train_labels)
    print("Training Class Counts: ", class_counts)
    # Calculate inverse frequencies
    class_weights = 1.0 / class_counts

    # Normalize weights (optional, but often helpful)
    class_weights = class_weights / class_weights.sum() * len(class_counts) 

    print("Training Class Weights:", class_weights)
    
    train_loader=None

    loader_gen = torch.Generator().manual_seed(seed) # for reproducibility of dataloader shuffling
    if balanced:
        # Assign weight to each sample based on its class
        sample_weights = class_weights[train_labels]

        sampler = WeightedRandomSampler(
            sample_weights, 
            len(sample_weights), 
            replacement=True, # Allows oversampling of minority classes
            generator=loader_gen
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=loader_gen)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=loader_gen)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, generator=loader_gen)

    return train_loader, val_loader, test_loader, class_weights


class EEGMemmapDataset(Dataset):
    def __init__(self, X_path, y_path, X_shape):
        """
        Memory-efficient EEG dataset using numpy memmap.
        
        Inputs:
        - X_path: path to memmap .dat file for EEG features (float32)
        - y_path: path to memmap .dat file for labels (int64)
        - X_shape: tuple, shape of X (num_samples, channels, timesteps)
        """
        self.X = np.memmap(X_path, dtype='int8', mode='r').reshape(X_shape)
        self.y = np.memmap(y_path, dtype='int64', mode='r')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # convert to float only for this batch
        x = torch.from_numpy(self.X[idx].copy()).float()  # 0/1 → float
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y


def prepare_training_data_memmap(X_path, y_path, X_shape, batch_size, balanced=True,
                                 train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                                 pin_memory=True, num_workers=2):
    """
    Prepares train/val/test DataLoaders using memory-mapped EEG arrays.
    """
    # Load dataset (memory-mapped)
    full_dataset = EEGMemmapDataset(X_path, y_path, X_shape)
    total_size = len(full_dataset)
    
    # Split indices
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    lengths = [train_size, val_size, test_size]
    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, lengths)


    train_labels = full_dataset.y[train_dataset.indices]  # memmap indexing, no copy
    class_counts = np.bincount(train_labels)
    print("Training Class Counts: ", class_counts)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print("Training Class Weights:", class_weights)

    if balanced:
        sample_weights = class_weights[train_labels].astype(np.float32)
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  sampler=sampler, pin_memory=pin_memory,
                                  num_workers=num_workers)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, pin_memory=pin_memory,
                                  num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, pin_memory=pin_memory,
                            num_workers=num_workers)

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, pin_memory=pin_memory,
                             num_workers=num_workers)
    
    class_weights = torch.from_numpy(class_weights).float()

    return train_loader, val_loader, test_loader, class_weights