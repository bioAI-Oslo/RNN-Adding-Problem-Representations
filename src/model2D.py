import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, functional
import torch.optim as optim

import sys
sys.path.append('../src')
from datagen import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU: ", device)
else:
    device = torch.device("cpu")
    print("Running on CPU")

torch.set_default_device(device)

class RNN_circular_2D_xy(nn.Module):
    def __init__(self,input_size,hidden_size,lr=0.001,act_decay=0.01,irnn=True,outputnn=True,bias=True,Wx_normalize=False,activation=True,batch_size=64,nav_space=2):
        super().__init__()
        self.input_size = input_size
        # self.time_steps = time_steps
        self.hidden_size = hidden_size*nav_space
        self.nav_space = nav_space # Number of navigation dimensions, default 1D, if 2D we want torus

        self.hidden = torch.nn.Linear(hidden_size,hidden_size,bias=False)
        self.input = torch.nn.Linear(input_size,hidden_size,bias=bias)

        self.h0_layer = torch.nn.Linear(self.nav_space,hidden_size,bias=bias)

        self.outputnn = outputnn
        self.output = torch.nn.Linear(hidden_size,2*self.nav_space,bias=bias)

        self.act_decay = act_decay
        self.batch_size = batch_size
        self.base_training_tsteps = 20

        # Define activation functions
        self.activation = activation
        if self.activation:
            self.act = torch.nn.ReLU()
        else:
            self.act = torch.nn.Identity()

        # Initialize IRNN
        if irnn:
            torch.nn.init.eye_(self.hidden.weight)
            if bias:
                # torch.nn.init.constant_(self.hidden.bias, 0)
                torch.nn.init.constant_(self.input.bias, 0)

        if Wx_normalize:
        # Set norm of input weights to 1
            self.input.weight = torch.nn.Parameter(self.input.weight/torch.norm(self.input.weight,dim=0).unsqueeze(1),requires_grad=True)
            # Freeze weights
            self.input.weight.requires_grad = False
            # print('Wx_norm: ',torch.norm(self.input.weight,dim=0).unsqueeze(1))

        self.Wh_init = self.hidden.weight.detach().clone()
        self.Wx_init = self.input.weight.detach().clone()

        self.loss_func = torch.nn.MSELoss()

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.00)

        self.losses = []
        self.accs = []

    def forward(self, x, raw=False):
        # Make h0 trainable
        batch_size_forward = x.size(0)
        h = torch.zeros((batch_size_forward,self.hidden_size)) # Gives constant initial hidden state
        h[:,0] = 1
        h[:,self.hidden_size//2] = 1
        self.time_steps = x.size(1)
        # time_steps+1 because we want to include the initial hidden state
        self.hts = torch.zeros(self.time_steps+1, batch_size_forward, self.hidden_size)
        self.hts[0] = h
        # Main RNN loop
        for t in range(0,self.time_steps):
            h = self.act(self.hidden(h) + self.input(x[:,t,:]))
            self.hts[t+1] = h
        return self.hts

    def loss_fn(self, x, y_hat):
        y = self(x)
        # norm of hts at each time step regularized to be 1
        activity_L2 = self.act_decay*((torch.norm(y,dim=-1)-1)**2).sum()
        
        # Concatenate 0 (h0) to y_hat to make it the same size as y
        # WE HAVE TO CONCATENATE THE MIDDLE VALUE OF THE LABELS/WHERE THE LABELS START WHCIH IS OFTEN 0 OR pi
        y_hat = torch.cat((torch.ones(y_hat.size(0),1)*np.pi,y_hat),dim=1)
        # Permute y_hat to make it the same size as y
        y_hat = y_hat.permute(1,0)
        # angle_loss = 0
        
        # Main angle loss loop, checks difference in angles for multiple time steps back in time
        i = torch.arange(1, self.time_steps).unsqueeze(1)
        # Check for 40% of the time steps back in time, to avoid angles beeing too large (above pi) for acos so that they become ambiguous (acos pi-0.1 = acos pi+0.1)
        j = torch.arange(1, self.time_steps//2-int(self.time_steps*0.1)).unsqueeze(0)
        # THIS IS VERY UNCERTAIN, TRY >= AND >
        mask = i >= j
        j = j * mask
        # i = i * mask
        # Convert i and j to int
        normalizer = 1 / (torch.norm(y[i], dim=-1) * torch.norm(y[i-j], dim=-1))
        # Cant clamp between -1 and 1 because it will cause NaNs in training
        angle_test = torch.abs(torch.acos(torch.clamp(torch.sum(y[i]*y[i-j], dim=-1) * normalizer, -0.9999999, 0.9999999)))
        # Make the angles that are not supposed to be checked 0
        angle_test = angle_test * mask.unsqueeze(-1)
        # Must use torch.abs because the angle can be negative, but the angle_test only returns positive angles
        angle_theoretical = torch.abs(y_hat[i]-y_hat[i-j])
        # Make 0 and 2pi the same angle
        # angle_theoretical = torch.min(torch.remainder(2*np.pi-(y_hat[i]-y_hat[i-j]),2*np.pi),torch.remainder(2*np.pi-(y_hat[i-j]-y_hat[i]),2*np.pi))

        # Scale the loss so that the latter time steps dont have a larger loss than the earlier time steps
        mask_dim2 = mask.shape[1]
        mask_loss_scale = mask_dim2/mask.sum(dim=1).unsqueeze(1)
        angle_loss = torch.mean(((angle_test-angle_theoretical)*mask_loss_scale.unsqueeze(-1))**2)
        # angle_loss = torch.mean(((angle_test-angle_theoretical))**2)

        self.losses.append(angle_loss.item())
        self.losses_norm.append(activity_L2.item())
        loss = angle_loss + activity_L2 # + circle_end_loss
        return loss

    def train_step(self, x, y_hat):
        self.optimizer.zero_grad(set_to_none=True)
        # Backward hook to clip gradients
        loss = self.loss_fn(x, y_hat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 2.0) # Clip gradients as in paper Low et al.
        self.optimizer.step()
        # Print weight gradient norms
        # print("Hidden weight grad norm:",torch.norm(self.hidden.weight.grad))
        return loss.item()

    def plot_losses(self,average=None):
        losses = np.array(self.losses)
        losses_norm = np.array(self.losses_norm)
        if average == None:
            plt.plot(losses[10:],label="Angle Loss")
            plt.plot(losses_norm[10:], label="Activity norm Loss")
            plt.plot(losses[10:]+losses_norm[10:], label="Total Loss")
        else:
            if len(losses)%average != 0:
                losses = losses[:-(len(losses)%average)]
                losses_norm = losses_norm[:-(len(losses_norm)%average)]
                print("Losses array was not a multiple of average. Truncated to",len(losses))
            loss_data_avgd = losses.reshape(average,-1).mean(axis=1)
            loss_data_norm_avgd = losses_norm.reshape(average,-1).mean(axis=1)
            plt.plot(loss_data_avgd[3:], label="Angle Loss")
            plt.plot(loss_data_norm_avgd[3:], label="Activity norm Loss")
            plt.plot(loss_data_avgd[3:]+loss_data_norm_avgd[3:], label="Total Loss")
        plt.legend()
        plt.title("MSE Losses")
        plt.show()


    def train(self, epochs=100, loader=None):
        for epoch in tqdm(range(epochs)):
            data,labels = datagen_circular_pm(self.batch_size,self.base_training_tsteps,sigma=0.05)
            loss = self.train_step(data,labels)
        return self.losses
    
    def train_gradual(self, epochs=100, loader=None):
        i = 0
        training_steps = 1
        for epoch in tqdm(range(epochs)):
            if i%50 == 0:
                training_steps += 1
            data,labels = datagen_circular_pm(self.batch_size,training_steps,sigma=0.05)
            loss = self.train_step(data.to(device),labels.to(device))
            i+=1
        print("Last training time steps:",training_steps)
        return self.losses
