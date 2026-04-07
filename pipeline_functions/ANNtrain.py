import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset, WeightedRandomSampler



def train_ann(model, num_epochs, train_loader, val_loader, criterion, optimizer, scheduler, device,
              update_every=5, checkpoint_path=None):
    """
    Train an ANN model for EEG classification.

    Returns:
    - training_history dictionary with train/val loss, accuracy, and balanced accuracy
    """
    training_history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_bal_acc": [],
        "val_bal_acc": []
    }

    for e in range(num_epochs):
        train_loss, train_acc, train_counts, train_bal_acc = train_epoch_ann(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_counts, val_bal_acc = validate_ann(
            model, val_loader, criterion, device
        )

        training_history["train_loss"].append(train_loss)
        training_history["val_loss"].append(val_loss)
        training_history["train_acc"].append(train_acc)
        training_history["val_acc"].append(val_acc)
        training_history["train_bal_acc"].append(train_bal_acc)
        training_history["val_bal_acc"].append(val_bal_acc)

        if scheduler is not None:
            scheduler.step(val_loss)

        if (e + 1) % update_every == 0:
            if checkpoint_path is not None:
                checkpoint = f'{checkpoint_path}/ann_checkpoint_e{e+1}.tar'
                save_checkpoint_ann(epoch=e, model=model, optimizer=optimizer, scheduler=scheduler, loss=train_loss, history=training_history, path=checkpoint,)

            print(
                f"Epoch {e+1}: Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc*100:.2f}%, "
                f"Training Balanced Accuracy: {train_bal_acc*100:.2f}%,\n"
                f"    Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc*100:.2f}%, "
                f"Validation Balanced Accuracy: {val_bal_acc*100:.2f}%"
            )
            print(f"    Predictions for each class on Training Set: {train_counts}")
            print(f"    Predictions for each class on Val Set: {val_counts}")

    return training_history



def train_epoch_ann(model, train_loader, criterion, optimizer, device):
    total_loss = 0.0
    num_correct = 0
    total = 0

    TP, FN, TN, FP = 0, 0, 0, 0
    total_counts = torch.zeros(2, dtype=torch.long, device=device)

    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.size(0)

        pred = logits.argmax(dim=1)
        num_correct += (pred == y).sum().item()

        TP += ((pred == 1) & (y == 1)).sum().item()
        FN += ((pred == 0) & (y == 1)).sum().item()
        FP += ((pred == 1) & (y == 0)).sum().item()
        TN += ((pred == 0) & (y == 0)).sum().item()

        class_counts = torch.bincount(pred, minlength=2)
        total_counts += class_counts
        total += y.size(0)

    avg_loss = total_loss / total
    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    bal_acc = (sensitivity + specificity) / 2
    acc = num_correct / total

    return avg_loss, acc, total_counts, bal_acc



def validate_ann(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_correct = 0
    total = 0

    TP, FN, TN, FP = 0, 0, 0, 0
    total_counts = torch.zeros(2, dtype=torch.long, device=device)

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)

            pred = logits.argmax(dim=1)
            num_correct += (pred == y).sum().item()

            TP += ((pred == 1) & (y == 1)).sum().item()
            FN += ((pred == 0) & (y == 1)).sum().item()
            FP += ((pred == 1) & (y == 0)).sum().item()
            TN += ((pred == 0) & (y == 0)).sum().item()

            class_counts = torch.bincount(pred, minlength=2)
            total_counts += class_counts
            total += y.size(0)

    avg_loss = total_loss / total
    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    bal_acc = (sensitivity + specificity) / 2
    acc = num_correct / total

    return avg_loss, acc, total_counts, bal_acc



def test_ann(model, test_loader, criterion, device):
    """
    Final evaluation on the held-out test set.
    """
    model.eval()
    total_loss = 0.0
    num_correct = 0
    total = 0

    TP, FN, TN, FP = 0, 0, 0, 0
    total_counts = torch.zeros(2, dtype=torch.long, device=device)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)

            pred = logits.argmax(dim=1)
            num_correct += (pred == y).sum().item()

            TP += ((pred == 1) & (y == 1)).sum().item()
            FN += ((pred == 0) & (y == 1)).sum().item()
            FP += ((pred == 1) & (y == 0)).sum().item()
            TN += ((pred == 0) & (y == 0)).sum().item()

            class_counts = torch.bincount(pred, minlength=2)
            total_counts += class_counts
            total += y.size(0)

    avg_loss = total_loss / total
    sensitivity = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    bal_acc = (sensitivity + specificity) / 2
    acc = num_correct / total

    return avg_loss, acc, total_counts, bal_acc



def save_checkpoint_ann(epoch, model, optimizer, scheduler, loss, history, path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'history': history
    }

    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()

    torch.save(state, path)



def prepare_training_data_ann(X, y, batch_size, balanced=True, seed=42):
    """
    Same train/val/test split style as the current SNN code, but for ANN input.

    X can be:
    - (samples, channels, time)
    - (samples, features)

    The ANN model will flatten internally, so no manual flattening is required here.
    """
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()

    full_dataset = TensorDataset(X_tensor, y_tensor)

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    lengths = [train_size, val_size, test_size]
    g = torch.Generator().manual_seed(seed)

    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, lengths, generator=g)

    train_labels = y_tensor[train_dataset.indices]
    class_counts = torch.bincount(train_labels)
    print("Training Class Counts:", class_counts)

    class_weights = 1.0 / (class_counts.float() + 1e-8)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    print("Training Class Weights:", class_weights)

    loader_gen = torch.Generator().manual_seed(seed)

    if balanced:
        sample_weights = class_weights[train_labels]
        sampler = WeightedRandomSampler(
            sample_weights,
            len(sample_weights),
            replacement=True,
            generator=loader_gen,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, generator=loader_gen)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, class_weights


class EEGAnnMemmapDataset(Dataset):
    def __init__(self, X_path, y_path, X_shape, x_dtype='float32'):
        self.X = np.memmap(X_path, dtype=x_dtype, mode='r').reshape(X_shape)
        self.y = np.memmap(y_path, dtype='int64', mode='r')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx].copy()).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        return x, y
