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
    data = torch.cat((start_positions.unsqueeze(1),data),dim=1)
    data = data.unsqueeze(-1)
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

def rat_box(n_data,t_steps,speed_mean=5,speed_std=5,box_size=1):
    # Use rat-in-a-box package to generate data
    # Data: 2D velocity vectors
    data = torch.zeros((n_data, t_steps, 2))
    # Labels: 2D positions
    labels = torch.zeros((n_data, t_steps, 2))
    Env = Environment() 
    Ag = Agent(Env)
    # Ag.speed_mean = speed_mean
    # Ag.speed_std = speed_std
    for i in range(n_data):
        for j in range(t_steps): 
            data[i,j] = torch.tensor(Ag.pos)
            labels[i,j] = torch.tensor(Ag.velocity)
            Ag.update()
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