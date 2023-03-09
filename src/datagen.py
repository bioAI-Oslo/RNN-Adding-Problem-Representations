import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, functional
import torch.optim as optim

def datagen(n_data,t_steps,n_masks):
    data = torch.zeros((n_data, t_steps, 2))
    masks = torch.zeros((n_data, t_steps))
    mask_places = torch.randperm(t_steps)[:n_masks]
    for i in range(n_masks):
        masks[:, mask_places[i]] = 1
    values = torch.rand((n_data, t_steps)).round(decimals=2)
    data[:, :, 0] = masks
    data[:, :, 1] = values
    labels = torch.zeros(n_data)
    labels = torch.sum(data[:, :, 0]*data[:, :, 1], dim=1)
    return data,labels


def datagen_full_sum(n_data,t_steps,normalize=True):
    data = torch.zeros((n_data, t_steps))
    for i in range(n_data):
        # Rayleigh distribution for the random numbers to sum, scale=1
        data[i] = torch.tensor(np.random.rayleigh(0.1, t_steps))
        if normalize:
            data[i] = data[i]/sum(data[i])
    # To fit into model
    data = data.unsqueeze(2)
    labels = torch.zeros(n_data)
    labels = torch.sum(data, dim=1)
    labels = labels.squeeze()
    return data,labels

def datagen_full_sum_normal(n_data,t_steps,normalize=True):
    data = torch.rand((n_data, t_steps)).round(decimals=2)
    for i in range(n_data):
        # Normal distribution for random numbers to sum
        data[i] = torch.empty(t_steps).normal_(0.1,0.05)
        data[i] = abs(data[i])
        if normalize:
            data[i] = data[i]/sum(data[i])
    data = data.unsqueeze(2)
    labels = torch.zeros(n_data)
    labels = torch.sum(data, dim=1)
    labels = labels.squeeze()
    return data,labels

def datagen_timewise_labels(n_data,t_steps,n_masks):
    data = torch.zeros((n_data, t_steps, 2))
    masks = torch.zeros((n_data, t_steps))
    mask_places = torch.randperm(t_steps)[:n_masks]
    for i in range(n_masks):
        masks[:, mask_places[i]] = 1
    values = torch.rand((n_data, t_steps)).round(decimals=2)
    data[:, :, 0] = masks
    data[:, :, 1] = values
    labels = torch.zeros(n_data,t_steps)
    # labels = torch.sum(data[:, :, 0]*data[:, :, 1], dim=1)
    for i in range(n_data):
        for j in range(t_steps):
            labels[i,j] = torch.sum(data[i, :j+1, 0]*data[i, :j+1, 1])
    return data,labels