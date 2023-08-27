import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, functional
import torch.optim as optim

import ratinabox #IMPORT 
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

from datagen import datagen_lowetal

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU: ", device)
else:
    device = torch.device("cpu")
    print("Running on CPU")

torch.set_default_device(device)

@torch.compile
def ornstein_uhlenbeck(dt, x, drift=0.0, noise_scale=0.2, coherence_time=5.0):
    # From Ratinabox Github
    """An ornstein uhlenbeck process in x.
    x can be multidimensional
    Args:
        dt: update time step
        x: the stochastic variable being updated
        drift (float, or same type as x, optional): [description]. Defaults to 0.
        noise_scale (float, or same type as x, optional): Magnitude of deviations from drift. Defaults to 0.2 (20 cm s^-1 if units of x are in metres).
        coherence_time (float, optional):
        Effectively over what time scale you expect x to change. Can be a vector (one timescale for each element of x) directions. Defaults to 5.

    Returns:
        dx (same type as x); the required update to x
    """
    x = torch.Tensor(x)
    drift = drift * torch.ones_like(x)
    noise_scale = noise_scale * torch.ones_like(x)
    coherence_time = coherence_time * torch.ones_like(x)
    sigma = torch.sqrt((2 * noise_scale**2) / (coherence_time * dt))
    theta = 1 / coherence_time
    dx = theta * (drift - x) * dt + sigma * torch.randn(x.shape)*dt
    return dx

@torch.compile
def smooth_wandering_2D_squarefix(n_data,t_steps,bound=0.5,v_sigma=0.01,d_sigma=0.1,v_bound_reduction=0.15,stability=0.01):
    # Stability is the pertubation on the 90 degree turn in a boundry interaction, the higher the more stable but the paths will lie in a circle not go in the corners
    # Lower stability will fill the whole square with paths, but the path generation may be more unstable
    # v_bound_reduction is the reduction in velocity when hitting a boundry and a lower value here will increase the stability
    # Smooth wandering in 2D with small random pertubation on head direction and velocity
    # Save velocity in x and y direction in data
    data = torch.zeros((n_data, t_steps, 2))
    # Save position in x and y direction in labels
    labels = torch.zeros((n_data, t_steps, 2))
    start_directions = torch.rand(n_data).unsqueeze(1)*2*np.pi
    velocities = torch.tensor(np.random.rayleigh(v_sigma, (n_data,t_steps))) #torch.rand((n_data,t_steps))*v_sigma
    direction_pert = torch.randn((n_data,t_steps))*np.pi*d_sigma
    directions = torch.cumsum(direction_pert,dim=1)+start_directions
    data[:,:,0] = velocities*torch.cos(directions)
    data[:,:,1] = velocities*torch.sin(directions)
    labels[:,:,0] = torch.cumsum(data[:,:,0],dim=1)
    labels[:,:,1] = torch.cumsum(data[:,:,1],dim=1)
    bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
    count = 0
    while bound_mask.any():
        # Extract the first True in each row
        bound_mask_first_true = torch.zeros_like(bound_mask)
        first_true = torch.argmax(bound_mask.int(), dim=1)
        bound_mask_first_true[range(bound_mask.shape[0]), first_true] = True
        # Make sure if there is no True in a row, the first_true is False
        bound_mask_first_true = torch.logical_and(bound_mask, bound_mask_first_true)
        bound_mask[range(bound_mask.shape[0]), first_true] = False
        # If any of the positions are outside the bound, redraw the velocities and directions for those trajectories
        velocities[bound_mask_first_true] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask_first_true.sum(),)))*v_bound_reduction*0.999**count # torch.rand((bound_mask_first_true.sum(),))*v_sigma
        velocities[bound_mask] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask.sum(),))) # torch.rand((bound_mask.sum(),))*v_sigma
        # For the first timestep redraw for each trajectory the direction changes by 90 degrees
        direction_pert[bound_mask_first_true] = np.pi/2*int(np.sign(np.random.randint(0,2,(1,1))-0.5))*(torch.randn(1)*(stability)+1) # + torch.randn((bound_mask_first_true.sum(),))*np.pi*d_sigma
        direction_pert[bound_mask] = torch.randn((bound_mask.sum(),))*np.pi*d_sigma
        directions = torch.cumsum(direction_pert,dim=1)+start_directions
        data[:,:,0] = velocities*torch.cos(directions)
        data[:,:,1] = velocities*torch.sin(directions)
        labels[:,:,0] = torch.cumsum(data[:,:,0],dim=1)
        labels[:,:,1] = torch.cumsum(data[:,:,1],dim=1)
        bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
        count += 1
    # print(count)
    ### DATA IS NOT SCALED BY 2PI HERE AS IT SHOULD
    data = data.unsqueeze(-1)
    labels = labels*2*np.pi/(2*bound)
    return data, labels

@torch.compile
def smooth_wandering_2D_squarefix_hdv(n_data,t_steps,bound=0.5,v_sigma=0.01,d_sigma=0.1,v_bound_reduction=0.15,stability=0.01):
    # Data is headdirection, speed
    # Stability is the pertubation on the 90 degree turn in a boundry interaction, the higher the more stable but the paths will lie in a circle not go in the corners
    # Lower stability will fill the whole square with paths, but the path generation may be more unstable
    # v_bound_reduction is the reduction in velocity when hitting a boundry and a lower value here will increase the stability
    # Smooth wandering in 2D with small random pertubation on head direction and velocity
    # Save head direction and speed in data
    data = torch.zeros((n_data, t_steps, 2))
    # Save position in x and y direction in labels
    labels = torch.zeros((n_data, t_steps, 2))
    start_directions = torch.rand(n_data).unsqueeze(1)*2*np.pi
    velocities = torch.tensor(np.random.rayleigh(v_sigma, (n_data,t_steps))) #torch.rand((n_data,t_steps))*v_sigma
    direction_pert = torch.randn((n_data,t_steps))*np.pi*d_sigma
    directions = torch.cumsum(direction_pert,dim=1)+start_directions
    data[:,:,0] = torch.remainder(directions,2*np.pi)
    data[:,:,1] = velocities
    labels[:,:,0] = torch.cumsum(velocities*torch.cos(directions),dim=1)
    labels[:,:,1] = torch.cumsum(velocities*torch.sin(directions),dim=1)
    bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
    count = 0
    while bound_mask.any():
        # Extract the first True in each row
        bound_mask_first_true = torch.zeros_like(bound_mask)
        first_true = torch.argmax(bound_mask.int(), dim=1)
        bound_mask_first_true[range(bound_mask.shape[0]), first_true] = True
        # Make sure if there is no True in a row, the first_true is False
        bound_mask_first_true = torch.logical_and(bound_mask, bound_mask_first_true)
        bound_mask[range(bound_mask.shape[0]), first_true] = False
        # If any of the positions are outside the bound, redraw the velocities and directions for those trajectories
        velocities[bound_mask_first_true] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask_first_true.sum(),)))*v_bound_reduction*0.999**count # torch.rand((bound_mask_first_true.sum(),))*v_sigma
        velocities[bound_mask] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask.sum(),))) # torch.rand((bound_mask.sum(),))*v_sigma
        # For the first timestep redraw for each trajectory the direction changes by 90 degrees
        direction_pert[bound_mask_first_true] = np.pi/2*int(np.sign(np.random.randint(0,2,(1,1))-0.5))*(torch.randn(1)*(stability)+1) # + torch.randn((bound_mask_first_true.sum(),))*np.pi*d_sigma
        direction_pert[bound_mask] = torch.randn((bound_mask.sum(),))*np.pi*d_sigma
        directions = torch.cumsum(direction_pert,dim=1)+start_directions
        data[:,:,0] = torch.remainder(directions,2*np.pi)
        data[:,:,1] = velocities
        labels[:,:,0] = torch.cumsum(velocities*torch.cos(directions),dim=1)
        labels[:,:,1] = torch.cumsum(velocities*torch.sin(directions),dim=1)
        bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
        count += 1
    # print(count)
    ### DATA IS NOT SCALED BY 2PI HERE AS IT SHOULD
    data = data.unsqueeze(-1)
    labels = labels*2*np.pi/(2*bound)
    return data, labels

@torch.compile
def smooth_wandering_2D_squarefix_randomstart(n_data,t_steps,bound=0.5,v_sigma=0.01,d_sigma=0.1,v_bound_reduction=0.15,stability=0.01):
    # Stability is the pertubation on the 90 degree turn in a boundry interaction, the higher the more stable but the paths will lie in a circle not go in the corners
    # Lower stability will fill the whole square with paths, but the path generation may be more unstable
    # v_bound_reduction is the reduction in velocity when hitting a boundry and a lower value here will increase the stability
    # Smooth wandering in 2D with small random pertubation on head direction and velocity
    # Save velocity in x and y direction in data
    data = torch.zeros((n_data, t_steps, 2))
    # Save position in x and y direction in labels
    labels = torch.zeros((n_data, t_steps, 2))
    start_positions = torch.rand(n_data,2)*2*bound-bound
    # Draw start directions such that they are not pointing towards the boundry
    # Where start positions is in for example first quadrant, draw start directions in between pi and 3pi/2
    start_directions = torch.rand(n_data)
    # start_directions = torch.where(torch.logical_and(start_positions[:,0] < 0, start_positions[:,1] > 0), start_directions+np.pi/2, start_directions)
    start_directions = start_directions.unsqueeze(1)
    # start_directions = torch.rand(n_data).unsqueeze(1)*2*np.pi
    velocities = torch.tensor(np.random.rayleigh(v_sigma, (n_data,t_steps))) #torch.rand((n_data,t_steps))*v_sigma
    direction_pert = torch.randn((n_data,t_steps))*np.pi*d_sigma
    directions = torch.cumsum(direction_pert,dim=1)+start_directions
    data[:,:,0] = velocities*torch.cos(directions)
    data[:,:,1] = velocities*torch.sin(directions)
    labels[:,:,0] = torch.cumsum(data[:,:,0],dim=1) + start_positions[:,0].unsqueeze(1)
    labels[:,:,1] = torch.cumsum(data[:,:,1],dim=1) + start_positions[:,1].unsqueeze(1)
    bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
    count = 0
    while bound_mask.any():
        # Extract the first True in each row
        bound_mask_first_true = torch.zeros_like(bound_mask)
        # First true is the index of the first True in each row if there is one, else it is False
        first_true = torch.argmax(bound_mask.int(), dim=1)
        bound_mask_first_true[range(bound_mask.shape[0]), first_true] = True
        # Make sure if there is no True in a row, the first_true is False
        bound_mask_first_true = torch.logical_and(bound_mask, bound_mask_first_true)
        bound_mask[range(bound_mask.shape[0]), first_true] = False
        # Check if only one is True in each row
        # print(bound_mask_first_true.sum(dim=1) == 1)
        # If any of the positions are outside the bound, redraw the velocities and directions for those trajectories
        velocities[bound_mask_first_true] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask_first_true.sum(),)))*v_bound_reduction*0.999**count # torch.rand((bound_mask_first_true.sum(),))*v_sigma
        velocities[bound_mask] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask.sum(),))) # torch.rand((bound_mask.sum(),))*v_sigma
        # For the first timestep redraw for each trajectory the direction changes by 90 degrees
        direction_pert[bound_mask_first_true] = np.pi/2*int(np.sign(np.random.randint(0,2,(1,1))-0.5))*(torch.randn(1)*(stability)+1)
        direction_pert[bound_mask] = torch.randn((bound_mask.sum(),))*np.pi*d_sigma
        directions = torch.cumsum(direction_pert,dim=1)+start_directions
        data[:,:,0] = velocities*torch.cos(directions)
        data[:,:,1] = velocities*torch.sin(directions)
        labels[:,:,0] = torch.cumsum(data[:,:,0],dim=1) + start_positions[:,0].unsqueeze(1)
        labels[:,:,1] = torch.cumsum(data[:,:,1],dim=1) + start_positions[:,1].unsqueeze(1)
        bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
        count += 1
    # print(count)
    # Concatenate the start positions to the data
    start_positions = start_positions*2*np.pi/(2*bound)
    ### DATA IS NOT SCALED BY 2PI HERE AS IT SHOULD
    data = torch.cat((start_positions.unsqueeze(1),data),dim=1)
    data = data.unsqueeze(-1)
    labels = labels*2*np.pi/(2*bound)
    return data, labels

@torch.compile
def smooth_wandering_2D_squarefix_randomstart_hdv(n_data,t_steps,bound=0.5,v_sigma=0.01,d_sigma=0.1,v_bound_reduction=0.15,stability=0.01):
    # Data is headdirection, speed
    # Stability is the pertubation on the 90 degree turn in a boundry interaction, the higher the more stable but the paths will lie in a circle not go in the corners
    # Lower stability will fill the whole square with paths, but the path generation may be more unstable
    # v_bound_reduction is the reduction in velocity when hitting a boundry and a lower value here will increase the stability
    # Smooth wandering in 2D with small random pertubation on head direction and velocity
    # Save head direction and speed in data
    data = torch.zeros((n_data, t_steps, 2))
    # Save position in x and y direction in labels
    labels = torch.zeros((n_data, t_steps, 2))
    start_positions = torch.rand(n_data,2)*2*bound-bound
    # Draw start directions such that they are not pointing towards the boundry
    # Where start positions is in for example first quadrant, draw start directions in between pi and 3pi/2
    start_directions = torch.rand(n_data)
    # start_directions = torch.where(torch.logical_and(start_positions[:,0] < 0, start_positions[:,1] > 0), start_directions+np.pi/2, start_directions)
    start_directions = start_directions.unsqueeze(1)
    # start_directions = torch.rand(n_data).unsqueeze(1)*2*np.pi
    velocities = torch.tensor(np.random.rayleigh(v_sigma, (n_data,t_steps))) #torch.rand((n_data,t_steps))*v_sigma
    direction_pert = torch.randn((n_data,t_steps))*np.pi*d_sigma
    directions = torch.cumsum(direction_pert,dim=1)+start_directions
    data[:,:,0] = torch.remainder(directions,2*np.pi)
    data[:,:,1] = velocities
    labels[:,:,0] = torch.cumsum(velocities*torch.cos(directions),dim=1) + start_positions[:,0].unsqueeze(1)
    labels[:,:,1] = torch.cumsum(velocities*torch.sin(directions),dim=1) + start_positions[:,1].unsqueeze(1)
    bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
    count = 0
    while bound_mask.any():
        # Extract the first True in each row
        bound_mask_first_true = torch.zeros_like(bound_mask)
        # First true is the index of the first True in each row if there is one, else it is False
        first_true = torch.argmax(bound_mask.int(), dim=1)
        bound_mask_first_true[range(bound_mask.shape[0]), first_true] = True
        # Make sure if there is no True in a row, the first_true is False
        bound_mask_first_true = torch.logical_and(bound_mask, bound_mask_first_true)
        bound_mask[range(bound_mask.shape[0]), first_true] = False
        # Check if only one is True in each row
        # print(bound_mask_first_true.sum(dim=1) == 1)
        # If any of the positions are outside the bound, redraw the velocities and directions for those trajectories
        velocities[bound_mask_first_true] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask_first_true.sum(),)))*v_bound_reduction*0.999**count # torch.rand((bound_mask_first_true.sum(),))*v_sigma
        velocities[bound_mask] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask.sum(),))) # torch.rand((bound_mask.sum(),))*v_sigma
        # For the first timestep redraw for each trajectory the direction changes by 90 degrees
        direction_pert[bound_mask_first_true] = np.pi/2*int(np.sign(np.random.randint(0,2,(1,1))-0.5))*(torch.randn(1)*(stability)+1)
        direction_pert[bound_mask] = torch.randn((bound_mask.sum(),))*np.pi*d_sigma
        directions = torch.cumsum(direction_pert,dim=1)+start_directions
        data[:,:,0] = torch.remainder(directions,2*np.pi)
        data[:,:,1] = velocities
        labels[:,:,0] = torch.cumsum(velocities*torch.cos(directions),dim=1) + start_positions[:,0].unsqueeze(1)
        labels[:,:,1] = torch.cumsum(velocities*torch.sin(directions),dim=1) + start_positions[:,1].unsqueeze(1)
        bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
        count += 1
    # print(count)
    # Concatenate the start positions to the data
    start_positions = start_positions*2*np.pi/(2*bound)
    ### DATA IS NOT SCALED BY 2PI HERE AS IT SHOULD
    data = torch.cat((start_positions.unsqueeze(1),data),dim=1)
    data = data.unsqueeze(-1)
    labels = labels*2*np.pi/(2*bound)
    return data, labels



@torch.compile
def smooth_wandering_2D_squarefix_randomstart_hdv_vrng(n_data,t_steps,bound=0.5,v_sigma_mean=0.01,d_sigma=0.1,v_bound_reduction=0.15,stability=0.01):
    # Data is headdirection, speed
    # Stability is the pertubation on the 90 degree turn in a boundry interaction, the higher the more stable but the paths will lie in a circle not go in the corners
    # Lower stability will fill the whole square with paths, but the path generation may be more unstable
    # v_bound_reduction is the reduction in velocity when hitting a boundry and a lower value here will increase the stability
    # Smooth wandering in 2D with small random pertubation on head direction and velocity
    # Save head direction and speed in data
    data = torch.zeros((n_data, t_steps, 2))
    # Save position in x and y direction in labels
    labels = torch.zeros((n_data, t_steps, 2))
    start_positions = torch.rand(n_data,2)*2*bound-bound
    # Draw start directions such that they are not pointing towards the boundry
    # Where start positions is in for example first quadrant, draw start directions in between pi and 3pi/2
    start_directions = torch.rand(n_data)
    # start_directions = torch.where(torch.logical_and(start_positions[:,0] < 0, start_positions[:,1] > 0), start_directions+np.pi/2, start_directions)
    start_directions = start_directions.unsqueeze(1)
    # start_directions = torch.rand(n_data).unsqueeze(1)*2*np.pi
    # v_sigma = np.random.rayleigh(v_sigma_mean)
    v_sigma = np.random.rayleigh(v_sigma_mean, (n_data,1))
    velocities = torch.tensor(np.random.rayleigh(v_sigma, (n_data,t_steps))) #torch.rand((n_data,t_steps))*v_sigma
    direction_pert = torch.randn((n_data,t_steps))*np.pi*d_sigma
    directions = torch.cumsum(direction_pert,dim=1)+start_directions
    data[:,:,0] = torch.remainder(directions,2*np.pi)
    data[:,:,1] = velocities
    labels[:,:,0] = torch.cumsum(velocities*torch.cos(directions),dim=1) + start_positions[:,0].unsqueeze(1)
    labels[:,:,1] = torch.cumsum(velocities*torch.sin(directions),dim=1) + start_positions[:,1].unsqueeze(1)
    bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
    count = 0
    while bound_mask.any():
        # Extract the first True in each row
        bound_mask_first_true = torch.zeros_like(bound_mask)
        # First true is the index of the first True in each row if there is one, else it is False
        first_true = torch.argmax(bound_mask.int(), dim=1)
        bound_mask_first_true[range(bound_mask.shape[0]), first_true] = True
        # Make sure if there is no True in a row, the first_true is False
        bound_mask_first_true = torch.logical_and(bound_mask, bound_mask_first_true)
        bound_mask[range(bound_mask.shape[0]), first_true] = False
        # Check if only one is True in each row
        # print(bound_mask_first_true.sum(dim=1) == 1)
        # If any of the positions are outside the bound, redraw the velocities and directions for those trajectories
        v_sigma = np.random.rayleigh(v_sigma_mean, (bound_mask_first_true.sum(),))
        velocities[bound_mask_first_true] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask_first_true.sum())))*v_bound_reduction*0.999**count # torch.rand((bound_mask_first_true.sum(),))*v_sigma
        v_sigma = np.random.rayleigh(v_sigma_mean, (bound_mask.sum(),))
        velocities[bound_mask] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask.sum(),))) # torch.rand((bound_mask.sum(),))*v_sigma
        # For the first timestep redraw for each trajectory the direction changes by 90 degrees
        direction_pert[bound_mask_first_true] = np.pi/2*int(np.sign(np.random.randint(0,2,(1,1))-0.5))*(torch.randn(1)*(stability)+1)
        direction_pert[bound_mask] = torch.randn((bound_mask.sum(),))*np.pi*d_sigma
        directions = torch.cumsum(direction_pert,dim=1)+start_directions
        data[:,:,0] = torch.remainder(directions,2*np.pi)
        data[:,:,1] = velocities
        labels[:,:,0] = torch.cumsum(velocities*torch.cos(directions),dim=1) + start_positions[:,0].unsqueeze(1)
        labels[:,:,1] = torch.cumsum(velocities*torch.sin(directions),dim=1) + start_positions[:,1].unsqueeze(1)
        bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
        count += 1
    # print(count)
    # Concatenate the start positions to the data
    start_positions = start_positions*2*np.pi/(2*bound)
    ### DATA IS NOT SCALED BY 2PI HERE AS IT SHOULD
    data = torch.cat((start_positions.unsqueeze(1),data),dim=1)
    data = data.unsqueeze(-1)
    labels = labels*2*np.pi/(2*bound)
    return data, labels

from scipy import stats as stats

def ornstein_uhlenbeck(dt, x, drift=0.0, noise_scale=0.2, coherence_time=5.0):
    # From Ratinabox Github
    """An ornstein uhlenbeck process in x.
    x can be multidimensional
    Args:
        dt: update time step
        x: the stochastic variable being updated
        drift (float, or same type as x, optional): [description]. Defaults to 0.
        noise_scale (float, or same type as x, optional): Magnitude of deviations from drift. Defaults to 0.2 (20 cm s^-1 if units of x are in metres).
        coherence_time (float, optional):
        Effectively over what time scale you expect x to change. Can be a vector (one timescale for each element of x) directions. Defaults to 5.

    Returns:
        dx (same type as x); the required update to x
    """
    x = torch.Tensor(x)
    drift = drift * torch.ones_like(x)
    noise_scale = noise_scale * torch.ones_like(x)
    coherence_time = coherence_time * torch.ones_like(x)
    sigma = torch.sqrt((2 * noise_scale**2) / (coherence_time * dt))
    theta = 1 / coherence_time
    dx = theta * (drift - x) * dt + sigma * torch.randn(x.shape)*dt
    return dx

def normal_to_rayleigh(x, sigma=1):
    """Converts a normally distributed variable (mean 0, var 1) to a rayleigh distributed variable (sigma)"""
    x = torch.tensor(stats.norm.cdf(x.cpu()))  # norm to uniform
    x = sigma * torch.sqrt(-2 * torch.log(1 - x))  # uniform to rayleigh
    return x

def rayleigh_to_normal(x, sigma=1):
    """Converts a rayleigh distributed variable (sigma) to a normally distributed variable (mean 0, var 1)"""
    x = 1 - torch.exp(-(x**2) / (2 * sigma**2))  # rayleigh to uniform
    x = torch.clamp(x, min=1e-6, max=1 - 1e-6)
    x = torch.tensor(stats.norm.ppf(x.cpu()))  # uniform to normal
    return x

def smooth_wandering_2D_ratinabox(n_data,t_steps,v_mean=0.08,v_sigma=0.08,v_coherence=0.7,d_sigma=120 * (np.pi / 180),d_coherence=10.0,d_scaler=10,v_bound_reduction=0.15,dt=0.1):
    bound = 0.5
    v_mean = v_mean
    speed_coherence = v_coherence
    direction_pert_coherence = d_coherence
    direction_pert_std=  d_sigma
    # Save head direction and speed in data
    data = torch.zeros((n_data, t_steps+1, 2))
    # Save position in x and y direction in labels
    labels = torch.zeros((n_data, t_steps, 2))
    start_positions = torch.rand(n_data,2)*2*bound-bound
    data[:,0,:] = start_positions
    
    # Draw start directions such that they are not pointing towards the boundry
    # Where start positions is in for example first quadrant, draw start directions in between pi and 3pi/2
    directions = torch.rand(n_data)
    # start_directions = start_directions.unsqueeze(1)
    speeds = v_mean*torch.ones(n_data)
    direction_perts = torch.zeros(n_data) # = start rotational speeds
    for t in range(t_steps):
        direction_perts_new = ornstein_uhlenbeck(dt,direction_perts,drift=0,noise_scale=direction_pert_std,coherence_time=direction_pert_coherence)
        if torch.any(speeds==0):  # add tiny velocity in [1,0] direction to avoid nans
            speeds[speeds==0] = 1e-8
        normal_variable = rayleigh_to_normal(speeds, sigma=v_mean)
        new_normal_variable = normal_variable + ornstein_uhlenbeck(dt=dt,x=normal_variable,drift=0,noise_scale=1,coherence_time=speed_coherence)
        new_speeds = normal_to_rayleigh(new_normal_variable, sigma=v_mean)
        directions_new = directions + direction_perts_new*d_scaler*dt
        directions_new = torch.remainder(directions_new,2*np.pi)
        data[:,t+1,0] = directions_new
        data[:,t+1,1] = new_speeds
        if t == 0:
            labels[:,t,0] = start_positions[:,0] + new_speeds*torch.cos(directions_new)*dt
            labels[:,t,1] = start_positions[:,1] + new_speeds*torch.sin(directions_new)*dt
        else:
            labels[:,t,0] = labels[:,t-1,0] + new_speeds*torch.cos(directions_new)*dt
            labels[:,t,1] = labels[:,t-1,1] + new_speeds*torch.sin(directions_new)*dt
        bound_mask = (labels[:,t,0] > bound) | (labels[:,t,0] < -bound) | (labels[:,t,1] > bound) | (labels[:,t,1] < -bound)
        while bound_mask.any():
            # If any of the positions are outside the bound, redraw the velocities and directions for those trajectories
            direction_perts_new[bound_mask] = torch.Tensor(np.random.choice([-1,1],int(bound_mask.sum()))).to(device)*(torch.randn((int(bound_mask.sum()),))*0.5 + np.pi/2)
            if torch.any(speeds[bound_mask]==0):  # add tiny velocity in [1,0] direction to avoid nans
                speeds[bound_mask][speeds[bound_mask]==0] = 1e-8
            normal_variable = rayleigh_to_normal(speeds[bound_mask], sigma=v_mean)
            new_normal_variable = normal_variable + ornstein_uhlenbeck(dt=dt,x=normal_variable,drift=0,noise_scale=1,coherence_time=speed_coherence)
            new_speeds[bound_mask] = normal_to_rayleigh(new_normal_variable, sigma=v_mean)*v_bound_reduction
            directions_new[bound_mask] = directions[bound_mask] + direction_perts_new[bound_mask]
            directions_new[bound_mask] = torch.remainder(directions_new[bound_mask],2*np.pi)
            
            data[bound_mask,t+1,0] = directions_new[bound_mask]
            data[bound_mask,t+1,1] = new_speeds[bound_mask].float()
            if t == 0:
                labels[bound_mask,t,0] = (start_positions[bound_mask,0] + new_speeds[bound_mask]*torch.cos(directions_new[bound_mask])*dt).float()
                labels[bound_mask,t,1] = (start_positions[bound_mask,1] + new_speeds[bound_mask]*torch.sin(directions_new[bound_mask])*dt).float()
            else:
                labels[bound_mask,t,0] = (labels[bound_mask,t-1,0] + new_speeds[bound_mask]*torch.cos(directions_new[bound_mask])*dt).float()
                labels[bound_mask,t,1] = (labels[bound_mask,t-1,1] + new_speeds[bound_mask]*torch.sin(directions_new[bound_mask])*dt).float()
            bound_mask = (labels[:,t,0] > bound) | (labels[:,t,0] < -bound) | (labels[:,t,1] > bound) | (labels[:,t,1] < -bound)
        speeds = new_speeds
        directions = directions_new
        direction_perts = direction_perts_new
        

    data = data.unsqueeze(-1)
    # Scale by 2pi to be in range -pi to pi
    data[:,0,:,:] = data[:,0,:,:]*2*np.pi/(2*bound)
    data[:,1:,1,:] = data[:,1:,1,:] * 2*np.pi/(2*bound)
    labels = labels*2*np.pi/(2*bound)
    return data, labels

        

def smooth_wandering_2D_circular(n_data,t_steps,bound=0.5,v_sigma=0.1,d_sigma=0.1):
    # Smooth wandering where you come back to other side of bound like pac-man
    # Smooth wandering in 2D with small random pertubation on head direction and velocity
    # Save velocity in x and y direction in data
    data = torch.zeros((n_data, t_steps, 2))
    # Save position in x and y direction in labels
    labels = torch.zeros((n_data, t_steps, 2))
    start_directions = torch.rand(n_data).unsqueeze(1)*2*np.pi
    velocities = torch.tensor(np.random.rayleigh(v_sigma, (n_data,t_steps))) #torch.rand((n_data,t_steps))*v_sigma
    direction_pert = torch.randn((n_data,t_steps))*np.pi*d_sigma
    directions = torch.cumsum(direction_pert,dim=1)+start_directions
    data[:,:,0] = velocities*torch.cos(directions)
    data[:,:,1] = velocities*torch.sin(directions)
    labels[:,:,0] = torch.cumsum(data[:,:,0],dim=1)
    labels[:,:,1] = torch.cumsum(data[:,:,1],dim=1)
    # Bound the labels to be between -bound and bound
    labels[:,:,0] = torch.remainder(labels[:,:,0]+bound,2*bound)-bound
    labels[:,:,1] = torch.remainder(labels[:,:,1]+bound,2*bound)-bound
    data = data.unsqueeze(-1)
    labels = labels*2*np.pi/(2*bound)
    return data, labels

def smooth_wandering_2D_circular_hdv(n_data,t_steps,bound=0.5,v_sigma=0.1,d_sigma=0.1):
    # Smooth wandering where you come back to other side of bound like pac-man
    # Smooth wandering in 2D with small random pertubation on head direction and velocity
    # Save head direction and speed in data
    data = torch.zeros((n_data, t_steps, 2))
    # Save position in x and y direction in labels
    labels = torch.zeros((n_data, t_steps, 2))
    start_directions = torch.rand(n_data).unsqueeze(1)*2*np.pi
    velocities = torch.tensor(np.random.rayleigh(v_sigma, (n_data,t_steps))) #torch.rand((n_data,t_steps))*v_sigma
    direction_pert = torch.randn((n_data,t_steps))*np.pi*d_sigma
    directions = torch.cumsum(direction_pert,dim=1)+start_directions
    data[:,:,0] = torch.remainder(directions,2*np.pi)
    data[:,:,1] = velocities
    labels[:,:,0] = torch.cumsum(velocities*torch.cos(directions),dim=1)
    labels[:,:,1] = torch.cumsum(velocities*torch.sin(directions),dim=1)
    # Bound the labels to be between -bound and bound
    labels[:,:,0] = torch.remainder(labels[:,:,0]+bound,2*bound)-bound
    labels[:,:,1] = torch.remainder(labels[:,:,1]+bound,2*bound)-bound
    data = data.unsqueeze(-1)
    labels = labels*2*np.pi/(2*bound)
    return data, labels

def smooth_wandering_2D_circular_randomstart(n_data,t_steps,bound=0.5,v_sigma=0.1,d_sigma=0.1):
    # Smooth wandering where you come back to other side of bound like pac-man
    # Smooth wandering in 2D with small random pertubation on head direction and velocity
    # Save velocity in x and y direction in data
    data = torch.zeros((n_data, t_steps, 2))
    # Save position in x and y direction in labels
    labels = torch.zeros((n_data, t_steps, 2))
    start_positions = torch.rand((n_data,2))*2*bound-bound
    start_directions = torch.rand(n_data).unsqueeze(1)*2*np.pi
    velocities = torch.tensor(np.random.rayleigh(v_sigma, (n_data,t_steps))) #torch.rand((n_data,t_steps))*v_sigma
    direction_pert = torch.randn((n_data,t_steps))*np.pi*d_sigma
    directions = torch.cumsum(direction_pert,dim=1)+start_directions
    data[:,:,0] = velocities*torch.cos(directions)
    data[:,:,1] = velocities*torch.sin(directions)
    labels[:,:,0] = torch.cumsum(data[:,:,0],dim=1) + start_positions[:,0].unsqueeze(1)
    labels[:,:,1] = torch.cumsum(data[:,:,1],dim=1) + start_positions[:,1].unsqueeze(1)
    # Bound the labels to be between -bound and bound
    labels[:,:,0] = torch.remainder(labels[:,:,0]+bound,2*bound)-bound
    labels[:,:,1] = torch.remainder(labels[:,:,1]+bound,2*bound)-bound
    # Concatenate the start positions to the data
    start_positions = start_positions*2*np.pi/(2*bound)
    data = torch.cat((start_positions.unsqueeze(1),data),dim=1)
    data = data.unsqueeze(-1)
    labels = labels*2*np.pi/(2*bound)
    return data, labels

def smooth_wandering_2D_circular_randomstart_hdv(n_data,t_steps,bound=0.5,v_sigma=0.1,d_sigma=0.1):
    # Smooth wandering where you come back to other side of bound like pac-man
    # Smooth wandering in 2D with small random pertubation on head direction and velocity
    # Save heading direction and velocity in data
    data = torch.zeros((n_data, t_steps, 2))
    # Save position in x and y direction in labels
    labels = torch.zeros((n_data, t_steps, 2))
    start_positions = torch.rand((n_data,2))*2*bound-bound
    start_directions = torch.rand(n_data).unsqueeze(1)*2*np.pi
    velocities = torch.tensor(np.random.rayleigh(v_sigma, (n_data,t_steps))) #torch.rand((n_data,t_steps))*v_sigma
    direction_pert = torch.randn((n_data,t_steps))*np.pi*d_sigma
    directions = torch.cumsum(direction_pert,dim=1)+start_directions
    data[:,:,0] = torch.remainder(directions,2*np.pi)
    data[:,:,1] = velocities
    labels[:,:,0] = torch.cumsum(velocities*torch.cos(directions),dim=1) + start_positions[:,0].unsqueeze(1)
    labels[:,:,1] = torch.cumsum(velocities*torch.sin(directions),dim=1) + start_positions[:,1].unsqueeze(1)
    # Bound the labels to be between -bound and bound
    labels[:,:,0] = torch.remainder(labels[:,:,0]+bound,2*bound)-bound
    labels[:,:,1] = torch.remainder(labels[:,:,1]+bound,2*bound)-bound
    # Concatenate the start positions to the data
    start_positions = start_positions*2*np.pi/(2*bound)
    data = torch.cat((start_positions.unsqueeze(1),data),dim=1)
    data = data.unsqueeze(-1)
    labels = labels*2*np.pi/(2*bound)
    return data, labels

def lowetal_2D(n_data,t_steps):
    # Not a smooth path at all
    datax, labelsx, pathsx = datagen_lowetal(n_data,t_steps)
    datay, labelsy, pathsy = datagen_lowetal(n_data,t_steps)
    datax = datax.squeeze()
    datay = datay.squeeze()
    data = torch.zeros((n_data, t_steps, 2))
    # Save position in x and y direction in labels
    # labels = torch.zeros((n_data, t_steps, 2))
    paths = torch.zeros((n_data, t_steps, 2))
    data[:,:,0] = datax
    data[:,:,1] = datay
    paths[:,:,0] = pathsx
    paths[:,:,1] = pathsy
    data = data.unsqueeze(-1)
    return data, paths


def random_walk(n, dt=0.1, x0=0.0, y0=0.0, v0=0.0, sigma=1.0):
    """
    Generate a smooth random walk in a square bound.

    Parameters:
    n (int): Number of timesteps
    dt (float): Timestep size
    x0 (float): Initial x position
    y0 (float): Initial y position
    v0 (float): Initial velocity
    sigma (float): Standard deviation of the random force

    Returns:
    tuple: A tuple containing:
        - x (numpy.ndarray): Array of x positions
        - y (numpy.ndarray): Array of y positions
        - vx (numpy.ndarray): Array of x velocities
        - vy (numpy.ndarray): Array of y velocities
    """
    # Initialize arrays
    x = np.zeros(n)
    y = np.zeros(n)
    vx = np.zeros(n)
    vy = np.zeros(n)

    # Set initial conditions
    x[0] = x0
    y[0] = y0
    vx[0] = v0 * np.cos(np.random.rand() * 2 * np.pi)
    vy[0] = v0 * np.sin(np.random.rand() * 2 * np.pi)

    # Time integration
    for i in range(1, n):
        # Compute random force
        fx = sigma * np.random.randn()
        fy = sigma * np.random.randn()

        # Update velocity
        vx[i] = vx[i-1] + fx * dt
        vy[i] = vy[i-1] + fy * dt

        # Update position
        x[i] = x[i-1] + vx[i] * dt
        y[i] = y[i-1] + vy[i] * dt

        # Check for boundary conditions and reflect if necessary
        if x[i] < 0:
            x[i] = -x[i]
            vx[i] = -vx[i]
        elif x[i] > 1:
            x[i] = 2 - x[i]
            vx[i] = -vx[i]
        if y[i] < 0:
            y[i] = -y[i]
            vy[i] = -vy[i]
        elif y[i] > 1:
            y[i] = 2 - y[i]
            vy[i] = -vy[i]

    return x, y, vx, vy

@torch.compile
def rat_box(n_data,t_steps,speed_mean=5,speed_std=5,box_size=1):
    # Use rat-in-a-box package to generate data
    # Data: 2D velocity vectors
    data = torch.zeros((n_data, t_steps, 2))
    # Labels: 2D positions
    labels = torch.zeros((n_data, t_steps, 2))
    # Ag.speed_mean = speed_mean
    # Ag.speed_std = speed_std
    for i in tqdm(range(n_data)):
        Env = Environment() 
        Ag = Agent(Env)
        for j in range(t_steps): 
            data[i,j] = torch.tensor(Ag.pos)
            labels[i,j] = torch.tensor(Ag.velocity)
            Ag.update()
    data = data.unsqueeze(3)
    # labels = labels*box_size
    return data, labels

@torch.compile
def rat_box_v2(n_data,t_steps,hdv=True):
    # Use rat-in-a-box package to generate data
    # Data: 2D velocity vectors
    data = torch.zeros((n_data, t_steps+1, 2))
    # Labels: 2D positions
    labels = torch.zeros((n_data, t_steps, 2))
    # Ag.speed_mean = speed_mean
    # Ag.speed_std = speed_std
    # for i in tqdm(range(n_data)):
    for i in range(n_data):
        Env = Environment() 
        Ag = Agent(Env)
        Ag.dt = 0.1
        data[i,0] = torch.tensor(Ag.pos)*2*np.pi - np.pi
        for j in range(t_steps): 
            Ag.update()
        data[i,1:] = torch.tensor(Ag.history["vel"])*2*np.pi
        labels[i] = torch.tensor(Ag.history["pos"])*2*np.pi - np.pi
    if hdv:
        data_old = data.clone()
        data[:,1:,0] = torch.atan2(data_old[:,1:,0],data_old[:,1:,1])
        data[:,1:,1] = data_old[:,1:].norm(dim=-1)
    data = data.unsqueeze(3)
    # labels = labels*box_size
    return data, labels


            

def rat_box_vemund(n_data,t_steps,box_size=1):
    from ratsimulator import Agent, trajectory_generator, batch_trajectory_generator
    from ratsimulator.Environment import Rectangle

    # Init Environment
    boxsize = (box_size,box_size)
    soft_boundary = 0.03
    environment = Rectangle(boxsize=boxsize, soft_boundary=soft_boundary)
    # agent = Agent(environment, boundary_mode="sorchers")
    labels = torch.zeros((n_data,t_steps,2))
    data = torch.zeros((n_data,t_steps,2))

    for i in range(n_data):
        gen = trajectory_generator(environment,seq_len=t_steps-1)
        outputs = next(gen)
        labels[i], data[i] = torch.tensor(outputs[0]), torch.tensor(outputs[1])

    data = data.unsqueeze(-1)
    return data, labels

def sincos_from_2D(labels):
    # Convert 2D data and labels to sin and cos for each dimension
    labels_new = torch.zeros((labels.shape[0],labels.shape[1],labels.shape[2]*2))
    for i in range(labels.shape[2]):
        labels_new[:,:,2*i] = torch.sin(labels[:,:,i])
        labels_new[:,:,2*i+1] = torch.cos(labels[:,:,i])
    return labels_new

def sincos_to_2D(labels):
    # Convert sin and cos to 2D data and labels
    labels_new = torch.zeros((labels.shape[0],labels.shape[1],labels.shape[2]//2))
    for i in range(labels.shape[2]//2):
        labels_new[:,:,i] = torch.atan2(labels[:,:,2*i],labels[:,:,2*i+1])
    return labels_new

def hd_direction_input_convert(input):
    # hd is relative angle to angle to start position
    input_data = input.copy()
    for i in range(len(input_data)):
        path = input_data[i,1][:,:,:]
        data = input_data[i,0][:,:,:].squeeze()
        p0 = data[:,0,:]
        hd = data[:,1:,0]
        dist_vec = path - p0.unsqueeze(1)
        dist = torch.norm(dist_vec, dim=2, keepdim=True)
        dist_angle = torch.atan2(dist_vec[:,:,1], dist_vec[:,:,0])
        rel_angle = torch.minimum(torch.remainder(dist_angle - hd, 2*np.pi), torch.remainder(hd - dist_angle, 2*np.pi))
        # To make angle direction unique
        flip = torch.isclose(rel_angle,torch.remainder(dist_angle - hd, 2*np.pi))
        rel_angle = torch.where(flip, -rel_angle, rel_angle)
        input_data[i,1][:,:,0] = rel_angle
        input_data[i,1][:,:,1] = dist.squeeze(-1)
    return input_data

def hd_direction_input_convert_partial(data,labels):
    # hd is relative angle to angle to start position
    labels_new = labels.clone()
    path = labels_new
    data = data.squeeze()
    p0 = data[:,0,:]
    hd = data[:,1:,0]
    dist_vec = path - p0.unsqueeze(1)
    dist = torch.norm(dist_vec, dim=2, keepdim=True)
    dist_angle = torch.atan2(dist_vec[:,:,1], dist_vec[:,:,0])
    rel_angle = torch.minimum(torch.remainder(dist_angle - hd, 2*np.pi), torch.remainder(hd - dist_angle, 2*np.pi))
    # To make angle direction unique
    flip = torch.isclose(rel_angle,torch.remainder(dist_angle - hd, 2*np.pi))
    rel_angle = torch.where(flip, -rel_angle, rel_angle)
    labels_new[:,:,0] = rel_angle
    labels_new[:,:,1] = dist.squeeze(-1)
    return data.unsqueeze(-1),labels_new

def convert_23D(data,labels):
    # Convert 2D data to 0,60,120 degree basis
    basis1 = torch.tensor([np.cos(0), np.sin(0)]).to(device)
    basis2 = torch.tensor([np.cos(np.pi/3), np.sin(np.pi/3)]).to(device)
    basis3 = torch.tensor([np.cos(2*np.pi/3), np.sin(2*np.pi/3)]).to(device)
    basises = torch.stack([basis1, basis2, basis3], dim=0).to(device)
    # Scale the decompositioned data so that they can be reconstructed to the original data
    basis_scale = 2/3

    data_23D = torch.zeros(data.shape[0],data.shape[1],3).to(device)
    labels_23D = torch.zeros(labels.shape[0],labels.shape[1],3).to(device)
    for i,basis in enumerate(basises):
        data_23D[:,:,i] = torch.sum(data.squeeze().to(device)*basis.to(device),dim=2)*basis_scale
        labels_23D[:,:,i] = torch.sum(labels.to(device)*basis.to(device),dim=2)*basis_scale

    data_23D = data_23D.unsqueeze(-1)
    return data_23D, labels_23D

def convert_23D_single(input):
    # Convert 2D data to 0,60,120 degree basis
    basis1 = torch.tensor([np.cos(0), np.sin(0)]).to(device)
    basis2 = torch.tensor([np.cos(np.pi/3), np.sin(np.pi/3)]).to(device)
    basis3 = torch.tensor([np.cos(2*np.pi/3), np.sin(2*np.pi/3)]).to(device)
    basises = torch.stack([basis1, basis2, basis3], dim=0).to(device)
    # Scale the decompositioned data so that they can be reconstructed to the original data
    basis_scale = 2/3

    converted = torch.zeros(input.shape[0],3).to(device)
    for i,basis in enumerate(basises):
        converted[:,i] = torch.sum(input.squeeze(-1).to(device)*basis.to(device),dim=-1)*basis_scale

    if len(input.shape) == 3:
        converted = converted.unsqueeze(-1)
    return converted