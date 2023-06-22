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

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU: ", device)
else:
    device = torch.device("cpu")
    print("Running on CPU")

torch.set_default_device(device)

def smooth_wandering_2D(n_data,t_steps,bound=0.5,v_sigma=0.1,d_sigma=0.1):
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
    # Print if first position is outside the bound
    # print((labels[:,0,0] > bound) | (labels[:,0,0] < -bound) | (labels[:,0,1] > bound) | (labels[:,0,1] < -bound))
    bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
    while bound_mask.any():
        # If any of the positions are outside the bound, redraw the velocities and directions for those trajectories
        velocities[bound_mask] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask.sum(),))) # torch.rand((bound_mask.sum(),))*v_sigma
        # For the first timestep redraw for each trajectory the direction changes by 90 degrees
        direction_pert[bound_mask] = torch.randn((bound_mask.sum(),))*np.pi*d_sigma
        directions = torch.cumsum(direction_pert,dim=1)+start_directions
        data[:,:,0] = velocities*torch.cos(directions)
        data[:,:,1] = velocities*torch.sin(directions)
        labels[:,:,0] = torch.cumsum(data[:,:,0],dim=1)
        labels[:,:,1] = torch.cumsum(data[:,:,1],dim=1)
        bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
    data = data.unsqueeze(-1)
    return data, labels

# @torch.compile
def smooth_wandering_2D_complex_bound(n_data,t_steps,bound=0.5,v_sigma=0.1,d_sigma=0.1):
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
        bound_mask[range(bound_mask.shape[0]), first_true] = False
        # If any of the positions are outside the bound, redraw the velocities and directions for those trajectories
        velocities[bound_mask] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask.sum(),))) # torch.rand((bound_mask.sum(),))*v_sigma
        # For the first timestep redraw for each trajectory the direction changes by 90 degrees
        direction_pert[bound_mask_first_true] = torch.randn((bound_mask_first_true.sum(),))*np.pi + directions[bound_mask_first_true] + np.pi
        direction_pert[bound_mask] = torch.randn((bound_mask.sum(),))*np.pi*d_sigma
        directions = torch.cumsum(direction_pert,dim=1)+start_directions
        data[:,:,0] = velocities*torch.cos(directions)
        data[:,:,1] = velocities*torch.sin(directions)
        labels[:,:,0] = torch.cumsum(data[:,:,0],dim=1)
        labels[:,:,1] = torch.cumsum(data[:,:,1],dim=1)
        bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
        count += 1
    print(count)
    data = data.unsqueeze(-1)
    labels = labels*2*np.pi/(2*bound)
    return data, labels

def smooth_wandering_2D_squarefix(n_data,t_steps,bound=0.5,v_sigma=0.1,d_sigma=0.1,v_bound_reduction=0.15,stability=0.01):
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
        bound_mask[range(bound_mask.shape[0]), first_true] = False
        # If any of the positions are outside the bound, redraw the velocities and directions for those trajectories
        velocities[bound_mask_first_true] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask_first_true.sum(),)))*v_bound_reduction # torch.rand((bound_mask_first_true.sum(),))*v_sigma
        velocities[bound_mask] = torch.tensor(np.random.rayleigh(v_sigma, (bound_mask.sum(),))) # torch.rand((bound_mask.sum(),))*v_sigma
        # For the first timestep redraw for each trajectory the direction changes by 90 degrees
        direction_pert[bound_mask_first_true] = np.pi/2*int(np.sign(np.random.randint(0,2,(1,1))-0.5))*(torch.randn(1)*stability+1) # + torch.randn((bound_mask_first_true.sum(),))*np.pi*d_sigma
        direction_pert[bound_mask] = torch.randn((bound_mask.sum(),))*np.pi*d_sigma
        directions = torch.cumsum(direction_pert,dim=1)+start_directions
        data[:,:,0] = velocities*torch.cos(directions)
        data[:,:,1] = velocities*torch.sin(directions)
        labels[:,:,0] = torch.cumsum(data[:,:,0],dim=1)
        labels[:,:,1] = torch.cumsum(data[:,:,1],dim=1)
        bound_mask = (labels[:,:,0] > bound) | (labels[:,:,0] < -bound) | (labels[:,:,1] > bound) | (labels[:,:,1] < -bound)
        count += 1
    print(count)
    data = data.unsqueeze(-1)
    labels = labels*2*np.pi/(2*bound)
    return data, labels

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

def rat_box(n_data,t_steps,speed_mean=5,speed_std=5,box_size=1):
    # Data: 2D velocity vectors
    data = torch.zeros((n_data, t_steps, 2))
    # Labels: 2D positions
    labels = torch.zeros((n_data, t_steps, 2))
    Env = Environment() 
    Ag = Agent(Env)
    # Ag.speed_mean = speed_mean
    # Ag.speed_std = speed_std
    for i in tqdm(range(n_data)):
        for j in range(t_steps): 
            data[i,j] = torch.tensor(Ag.pos)
            labels[i,j] = torch.tensor(Ag.velocity)
            Ag.update()
    data = data.unsqueeze(3)
    # labels = labels*box_size
    return data, labels
            