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
    # For å generere data til generell adding problem
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
    # Datagen bare uten masker og med rayleigh distribusjon
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
    # Datagen bare uten masker og med normal distribusjon
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

def datagen_circular(n_data,t_steps):
    # Data for adding problem mapped to a circle, no masks, rayleigh distribution
    data = torch.zeros((n_data, t_steps))
    # VECTORIZE THIS
    for i in range(n_data):
        # Rayleigh distribution for the random numbers to sum, scale=1
        data[i] = torch.tensor(np.random.rayleigh(0.1, t_steps))
        # data[i][0] = 0
        data[i] = data[i]/sum(data[i])
    # Necessary to fit into nn model
    data = data.unsqueeze(2)
    # Labels are the angles of the points from 0 to 1 mapped on the circle
    labels = torch.zeros((n_data, t_steps))
    for i in range(n_data):
        for j in range(t_steps):
            # Map sums 0 to 1 to angles 0 to 2pi
            labels[i,j] = 2*np.pi*torch.sum(data[i,:j+1])
    # labels[:,0] = 0
    return data,labels

def datagen_circular_pm(n_data,t_steps,bound=0.5):
    # Data for adding/subtracting problem mapped to a circle, no masks, positive and negative values drawn from Gauss distribution
    # data = torch.zeros((n_data, t_steps))
    # labels = torch.zeros((n_data, t_steps))
    # for i in range(n_data):
    #     for j in range(t_steps):
    #         # Draw random number from Gauss distribution to add/subtract to sum
    #         data[i,j] = torch.empty(1).normal_(0,0.1)
    #         while torch.sum(data[i,:j+1]) > bound or torch.sum(data[i,:j+1]) < -bound:
    #             # If sum is outside of bounds, draw new number
    #             data[i,j] = torch.empty(1).normal_(0,0.1)
    #         # Timewise labels are the sums
    #         labels[i,j] = torch.sum(data[i,:j+1])
    # Set up data and labels tensors
    data = torch.randn((n_data, t_steps)) * 0.1
    cumsum = torch.cumsum(data, dim=1)
    mask = (cumsum > bound) | (cumsum < -bound)
    while mask.any():
        data[mask] = torch.randn((mask.sum(),)) * 0.1
        cumsum = torch.cumsum(data, dim=1)
        mask = (cumsum > bound) | (cumsum < -bound)
    labels = cumsum


    # Necessary to fit into nn model
    data = data.unsqueeze(2)
    # Map labels to angles 0 to 2pi
    labels = (labels)*2*np.pi/(2*bound) + np.pi
    return data,labels