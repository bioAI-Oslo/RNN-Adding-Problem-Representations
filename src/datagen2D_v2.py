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

from scipy import stats as stats

class datagen2D_OU():
    def __init__(self,randomstart=True,head_direction=True,circular=False,v_mean=0.08,v_sigma=0.08,v_coherence=0.7,d_sigma=120 * (np.pi / 180),d_coherence=10.0,d_scaler=10,v_bound_reduction=0.15,dt=0.1):
        self.v_mean = v_mean
        self.v_sigma = v_sigma
        self.v_coherence = v_coherence
        self.d_sigma = d_sigma
        self.d_coherence = d_coherence
        self.d_scaler = d_scaler
        self.v_bound_reduction = v_bound_reduction
        self.dt = dt
        self.randomstart = randomstart
        self.head_direction = head_direction
        self.circular = circular

        self.bound = 0.5

    def __call__(self,n_data,t_steps):
        if self.circular:
            return self.gen_data_circular(n_data,t_steps)
        else:
            return self.gen_data_bounded(n_data,t_steps)

    @staticmethod
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

    @staticmethod
    def normal_to_rayleigh(x, sigma=1):
        # From Ratinabox Github
        """Converts a normally distributed variable (mean 0, var 1) to a rayleigh distributed variable (sigma)"""
        x = torch.tensor(stats.norm.cdf(x.cpu()))  # norm to uniform
        x = sigma * torch.sqrt(-2 * torch.log(1 - x))  # uniform to rayleigh
        return x

    @staticmethod
    def rayleigh_to_normal(x, sigma=1):
        # From Ratinabox Github
        """Converts a rayleigh distributed variable (sigma) to a normally distributed variable (mean 0, var 1)"""
        x = 1 - torch.exp(-(x**2) / (2 * sigma**2))  # rayleigh to uniform
        x = torch.clamp(x, min=1e-6, max=1 - 1e-6)
        x = torch.tensor(stats.norm.ppf(x.cpu()))  # uniform to normal
        return x

    @staticmethod
    def head_direction_to_xy(head_direction,v):
        """Converts a head direction and speed to a velocity vector"""
        x = v * torch.cos(head_direction)
        y = v * torch.sin(head_direction)
        return x, y

    def gen_data_bounded(self,n_data,t_steps):
        bound = self.bound
        v_mean = self.v_mean
        speed_coherence = self.v_coherence
        direction_pert_coherence = self.d_coherence
        direction_pert_std=  self.d_sigma
        # Save head direction and speed in data
        data = torch.zeros((n_data, t_steps+1, 2))
        # Save position in x and y direction in labels
        labels = torch.zeros((n_data, t_steps, 2))
        if self.randomstart:
            start_positions = torch.rand(n_data,2)*2*bound-bound
        else:
            start_positions = torch.zeros(n_data,2)
        data[:,0,:] = start_positions
        directions = torch.rand(n_data)
        speeds = v_mean*torch.ones(n_data)
        direction_perts = torch.zeros(n_data)
        for t in range(t_steps):
            direction_perts_new = self.ornstein_uhlenbeck(self.dt,direction_perts,drift=0,noise_scale=direction_pert_std,coherence_time=direction_pert_coherence)
            if torch.any(speeds==0):
                speeds[speeds==0] = 1e-8
            normal_variable = self.rayleigh_to_normal(speeds, sigma=v_mean)
            new_normal_variable = normal_variable + self.ornstein_uhlenbeck(self.dt,x=normal_variable,drift=0,noise_scale=1,coherence_time=speed_coherence)
            new_speeds = self.normal_to_rayleigh(new_normal_variable, sigma=v_mean)
            directions_new = directions + direction_perts_new*self.d_scaler*self.dt
            directions_new = torch.remainder(directions_new,2*np.pi)
            data[:,t+1,0] = directions_new
            data[:,t+1,1] = new_speeds
            if t == 0:
                labels[:,t,0] = start_positions[:,0] + new_speeds*torch.cos(directions_new)*self.dt
                labels[:,t,1] = start_positions[:,1] + new_speeds*torch.sin(directions_new)*self.dt
            else:
                labels[:,t,0] = labels[:,t-1,0] + new_speeds*torch.cos(directions_new)*self.dt
                labels[:,t,1] = labels[:,t-1,1] + new_speeds*torch.sin(directions_new)*self.dt
            bound_mask = (labels[:,t,0] > bound) | (labels[:,t,0] < -bound) | (labels[:,t,1] > bound) | (labels[:,t,1] < -bound)
            while bound_mask.any():
                # If any of the positions are outside the bound, redraw the velocities and directions for those trajectories
                direction_perts_new[bound_mask] = torch.Tensor(np.random.choice([-1,1],int(bound_mask.sum()))).to(device)*(torch.randn((int(bound_mask.sum()),))*0.5 + np.pi/2)
                if torch.any(speeds[bound_mask]==0):
                    speeds[bound_mask][speeds[bound_mask]==0] = 1e-8
                normal_variable = self.rayleigh_to_normal(speeds[bound_mask], sigma=v_mean)
                new_normal_variable = normal_variable + self.ornstein_uhlenbeck(self.dt,x=normal_variable,drift=0,noise_scale=1,coherence_time=speed_coherence)
                new_speeds[bound_mask] = self.normal_to_rayleigh(new_normal_variable, sigma=v_mean)*self.v_bound_reduction
                directions_new[bound_mask] = directions[bound_mask] + direction_perts_new[bound_mask]
                directions_new[bound_mask] = torch.remainder(directions_new[bound_mask],2*np.pi)
                
                data[bound_mask,t+1,0] = directions_new[bound_mask]
                data[bound_mask,t+1,1] = new_speeds[bound_mask].float()
                if t == 0:
                    labels[bound_mask,t,0] = (start_positions[bound_mask,0] + new_speeds[bound_mask]*torch.cos(directions_new[bound_mask])*self.dt).float()
                    labels[bound_mask,t,1] = (start_positions[bound_mask,1] + new_speeds[bound_mask]*torch.sin(directions_new[bound_mask])*self.dt).float()
                else:
                    labels[bound_mask,t,0] = (labels[bound_mask,t-1,0] + new_speeds[bound_mask]*torch.cos(directions_new[bound_mask])*self.dt).float()
                    labels[bound_mask,t,1] = (labels[bound_mask,t-1,1] + new_speeds[bound_mask]*torch.sin(directions_new[bound_mask])*self.dt).float()
                bound_mask = (labels[:,t,0] > bound) | (labels[:,t,0] < -bound) | (labels[:,t,1] > bound) | (labels[:,t,1] < -bound)
            speeds = new_speeds
            directions = directions_new
            direction_perts = direction_perts_new
        data = data.unsqueeze(-1)
        # Scale by 2pi to be in range -pi to pi
        data[:,0,:,:] = data[:,0,:,:]*2*np.pi/(2*bound)
        data[:,1:,1,:] = data[:,1:,1,:] * 2*np.pi/(2*bound)
        if not self.head_direction:
            data[:,1:,0],data[:,1:,1] = self.head_direction_to_xy(data[:,1:,0],data[:,1:,1])
        labels = labels*2*np.pi/(2*bound)
        self.data = data
        self.labels = labels
        return data, labels
    
    def gen_data_circular(self,n_data,t_steps):
        # Same as above but with circular boundary conditions
        bound = self.bound
        v_mean = self.v_mean
        speed_coherence = self.v_coherence
        direction_pert_coherence = self.d_coherence
        direction_pert_std=  self.d_sigma
        # Save head direction and speed in data
        data = torch.zeros((n_data, t_steps+1, 2))
        # Save position in x and y direction in labels
        labels = torch.zeros((n_data, t_steps, 2))
        if self.randomstart:
            start_positions = torch.rand(n_data,2)*2*bound-bound
        else:
            start_positions = torch.zeros(n_data,2)
        data[:,0,:] = start_positions
        directions = torch.rand(n_data)
        # start_directions = start_directions.unsqueeze(1)
        speeds = v_mean*torch.ones(n_data)
        direction_perts = torch.zeros(n_data)
        for t in range(t_steps):
            direction_perts_new = self.ornstein_uhlenbeck(self.dt,direction_perts,drift=0,noise_scale=direction_pert_std,coherence_time=direction_pert_coherence)
            if torch.any(speeds==0):
                speeds[speeds==0] = 1e-8
            normal_variable = self.rayleigh_to_normal(speeds, sigma=v_mean)
            new_normal_variable = normal_variable + self.ornstein_uhlenbeck(self.dt,x=normal_variable,drift=0,noise_scale=1,coherence_time=speed_coherence)
            new_speeds = self.normal_to_rayleigh(new_normal_variable, sigma=v_mean)
            directions_new = directions + direction_perts_new*self.d_scaler*self.dt
            directions_new = torch.remainder(directions_new,2*np.pi)
            data[:,t+1,0] = directions_new
            data[:,t+1,1] = new_speeds
            if t == 0:
                labels[:,t,0] = start_positions[:,0] + new_speeds*torch.cos(directions_new)*self.dt
                labels[:,t,1] = start_positions[:,1] + new_speeds*torch.sin(directions_new)*self.dt
            else:
                labels[:,t,0] = labels[:,t-1,0] + new_speeds*torch.cos(directions_new)*self.dt
                labels[:,t,1] = labels[:,t-1,1] + new_speeds*torch.sin(directions_new)*self.dt
            # Remainder to keep positions in range 0 to 2pi
            speeds = new_speeds
            directions = directions_new
            direction_perts = direction_perts_new

        labels[:,:,0] = torch.remainder(labels[:,:,0]+bound,2*bound) - bound
        labels[:,:,1] = torch.remainder(labels[:,:,1]+bound,2*bound) - bound
        data = data.unsqueeze(-1)
        # Scale by 2pi to be in range -pi to pi
        data[:,0,:,:] = data[:,0,:,:]*2*np.pi/(2*bound)
        data[:,1:,1,:] = data[:,1:,1,:] * 2*np.pi/(2*bound)
        if not self.head_direction:
            data[:,1:,0],data[:,1:,1] = self.head_direction_to_xy(data[:,1:,0],data[:,1:,1])
            labels[:,:,0],labels[:,:,1] = self.head_direction_to_xy(labels[:,:,0],labels[:,:,1])
        labels = labels*2*np.pi/(2*bound)
        self.data = data
        self.labels = labels
        return data, labels







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
    # From Ratinabox Github
    """Converts a normally distributed variable (mean 0, var 1) to a rayleigh distributed variable (sigma)"""
    x = torch.tensor(stats.norm.cdf(x.cpu()))  # norm to uniform
    x = sigma * torch.sqrt(-2 * torch.log(1 - x))  # uniform to rayleigh
    return x

def rayleigh_to_normal(x, sigma=1):
    # From Ratinabox Github
    """Converts a rayleigh distributed variable (sigma) to a normally distributed variable (mean 0, var 1)"""
    x = 1 - torch.exp(-(x**2) / (2 * sigma**2))  # rayleigh to uniform
    x = torch.clamp(x, min=1e-6, max=1 - 1e-6)
    x = torch.tensor(stats.norm.ppf(x.cpu()))  # uniform to normal
    return x