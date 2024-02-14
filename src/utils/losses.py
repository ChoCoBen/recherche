import torch.nn as nn
import numpy as np
import torch

def compute_losses(prediction, target, test):
    criterion = nn.CrossEntropyLoss()
    acc = calculate_accuracy(nn.functional.softmax(prediction, dim=1), target)
    conf = calculate_confidence(nn.functional.softmax(prediction, dim=1))
    if not test:
        return {
            'total_loss_train': criterion(prediction, target),
            'train_accuracy': acc,
            'train_confidence': conf
        }
    else:
        return {
            'total_loss_test': criterion(prediction, target),
            'test_accuracy': acc,
            'test_confidence': conf
        }

def append_losses(losses, loss_history):

    if loss_history is None:
        loss_history = {}
        for k,v in losses.items():
            loss_history[k] = [v]
    
    else:
        for k,v in losses.items():
            loss_history[k].append(v)
    
    return loss_history

def average_losses(loss_history):
    averaged_losses = {}
    for k,v in loss_history.items():
        averaged_losses[k] = (float(np.mean(v)), float(np.std(v)))
    
    return averaged_losses

def display_losses(averaged_losses):
    print('[INFO] Average losses on test set')
    for k,v in averaged_losses.items():
        print(f'[INFO] \t {k.ljust(30)} : mean = {f"{v[0]:.2e}".ljust(10)} standard deviation = {f"{v[1]:.2e}".ljust(10)}')

def set_losses_wandb(averaged_losses):
    wandb_losses = {}
    for k,v in averaged_losses.items():
        wandb_losses[k] = v[0]
    return wandb_losses

def calculate_accuracy(prediction, target):
    gesture_pred = torch.argmax(prediction, dim=1)
    return torch.sum(gesture_pred == target).item() / target.shape[0]

def calculate_confidence(prediction):
    prediction_confidence = torch.max(prediction, dim=1).values
    return torch.mean(prediction_confidence)
    
