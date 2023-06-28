import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, functional
import torch.optim as optim

from Sophia import SophiaG

import sys
sys.path.append('../src')
from datagen import *
from datagen2D import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU: ", device)
else:
    device = torch.device("cpu")
    print("Running on CPU")

torch.set_default_device(device)

class RNN_circular_2D_xy_Low(nn.Module):
    def __init__(self,input_size,hidden_size,lr=0.0005,act_decay=0.0,weight_decay=0.01,irnn=True,outputnn=True,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=2):
        super().__init__()
        self.input_size = input_size
        # self.time_steps = time_steps
        self.hidden_size = hidden_size*nav_space
        self.nav_space = nav_space # Number of navigation dimensions, default 1D, if 2D we want torus

        self.hidden = torch.nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.inputx = torch.nn.Linear(input_size,self.hidden_size,bias=bias)
        self.inputy = torch.nn.Linear(input_size,self.hidden_size,bias=bias)

        self.h0_layer = torch.nn.Linear(self.nav_space,self.hidden_size,bias=bias)

        self.outputnn = outputnn
        self.output = torch.nn.Linear(self.hidden_size,2*self.nav_space,bias=bias) # Decoder from Low et al., 2*nav space since it decodes into cos and sin for every dimension

        self.act_decay = act_decay
        self.weight_decay = weight_decay
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
        self.Wx_x_init = self.inputx.weight.detach().clone()
        self.Wx_y_init = self.inputy.weight.detach().clone()

        self.loss_func = torch.nn.MSELoss()

        self.lr = lr
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.optimizer = SophiaG(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        self.losses = []
        self.accs = []

    def forward(self, x, raw=False):
        # Make h0 trainable
        batch_size_forward = x.size(0)
        h = torch.zeros((batch_size_forward,self.hidden_size)) # Gives constant initial hidden state
        # h = h/np.sqrt(self.hidden_size) # Normalize initial hidden state
        h[:,0] = 1 # Initialize first cell to 1
        self.time_steps = x.size(1)
        # time_steps+1 because we want to include the initial hidden state
        self.hts = torch.zeros(self.time_steps+1, batch_size_forward, self.hidden_size)
        self.hts[0] = h
        # Main RNN loop
        for t in range(0,self.time_steps):
            # print(x.shape)
            h = self.act(self.hidden(h) + self.inputx(x[:,t,0,:]) + self.inputy(x[:,t,1,:]))
            self.hts[t+1] = h
        if not raw:
            return self.output(self.hts)
        return self.hts

    def loss_fn(self, x, y_hat):
        y = self(x)[1:,:,:]
        # Activity loss
        activity_L2 = self.act_decay*((torch.norm(y,dim=-1)-1)**2).sum()
        y_hat = y_hat.transpose(0,1)
        loss_sin_x = self.loss_func(y[:,:,0],y_hat[:,:,0])
        loss_cos_x = self.loss_func(y[:,:,1],y_hat[:,:,1])
        loss_sin_y = self.loss_func(y[:,:,2],y_hat[:,:,2])
        loss_cos_y = self.loss_func(y[:,:,3],y_hat[:,:,3])
        loss = loss_sin_x + loss_cos_x + loss_sin_y + loss_cos_y + activity_L2
        self.losses.append(loss.item())
        return loss

    def train_step(self, x, y_hat):
        self.optimizer.zero_grad(set_to_none=True)
        # Backward hook to clip gradients
        loss = self.loss_fn(x, y_hat)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0) # Clip gradients as in paper Low et al.
        self.optimizer.step()
        # Print weight gradient norms
        # print("Hidden weight grad norm:",torch.norm(self.hidden.weight.grad))
        return loss.item()

    def plot_losses(self,average=None):
        losses = np.array(self.losses)
        if average == None:
            plt.plot(losses[10:])
        else:
            if len(losses)%average != 0:
                losses = losses[:-(len(losses)%average)]
                print("Losses array was not a multiple of average. Truncated to",len(losses))
            loss_data_avgd = losses.reshape(average,-1).mean(axis=1)
            plt.plot(loss_data_avgd[3:])
        plt.title("MSE Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()


    def train(self, epochs=100, loader=None):
        for epoch in tqdm(range(epochs)):
            # data,labels = datagen_circular_pm(self.batch_size,self.base_training_tsteps,sigma=0.05)
            data,labels = smooth_wandering_2D_squarefix(self.batch_size,self.base_training_tsteps)
            labels = sincos_from_2D(labels)
            loss = self.train_step(data.to(device),labels.to(device))
        return self.losses
    
    def train_gradual(self, epochs=100, loader=None):
        i = 0
        training_steps = 1
        for epoch in tqdm(range(epochs)):
            if i%15 == 0:
                training_steps += 1
            data,labels = smooth_wandering_2D_squarefix(self.batch_size,training_steps)
            labels = sincos_from_2D(labels)
            loss = self.train_step(data.to(device),labels.to(device))
            i+=1
        print("Last training time steps:",training_steps)
        return self.losses
    
    def train_gradual_manual(self,input):
        # Input shape: [Epochs,data/labels,batchsize,tsteps,x/y]
        for i in tqdm(range(len(input))):
            data = input[i][0]
            labels = input[i][1]
            labels = sincos_from_2D(labels)
            loss = self.train_step(data.to(device),labels.to(device))

class RNN_circular_2D_xy_relative(RNN_circular_2D_xy_Low):
    def __init__(self,input_size,hidden_size,lr=0.0005,act_decay=0.0,weight_decay=0.01,irnn=True,outputnn=True,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=2):
        super().__init__(input_size,hidden_size,lr=lr,act_decay=act_decay,weight_decay=weight_decay,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space)
        self.output = torch.nn.Linear(self.hidden_size,self.nav_space,bias=bias) # Decoder from hidden states to x and y
        self.Wx_out_init = self.output.weight.detach().clone()

        self.optimizer = SophiaG(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def loss_fn(self, x, y_hat):
        y = self(x,raw=True)
        # Activity loss
        activity_L2 = self.act_decay*((torch.norm(y,dim=-1)-1)**2).sum()
        y = self.output(y)
        # print(y_hat[:,:,0].shape, torch.ones(y_hat.size(0),1).shape)
        y_hat = torch.cat((torch.ones(y_hat.size(0),1,2)*0,y_hat),dim=1)
        # y_hat[:,:,1] = torch.cat((torch.ones(y_hat.size(0),1)*y_hat_start,y_hat[:,:,1]),dim=1)
        y_hat = y_hat.transpose(0,1)
        # Main relative angle diff loss loop, checks difference in position for multiple time steps back in time
        i = torch.arange(1, self.time_steps).unsqueeze(1)
        # Check for 40% of the time steps back in time, to reduce the number of comparisons (short term memory)
        j = torch.arange(1, max(self.time_steps//2-int(self.time_steps*0.1),1)).unsqueeze(0)
        mask = i >= j
        j = j * mask

        # print(i)
        # print(j)
        pred_diffs_x = y[i,:,0] - y[i-j,:,0]
        pred_diffs_y = y[i,:,1] - y[i-j,:,1]
        # Make the pos diffs that are not supposed to be checked 0
        
        pred_diffs_x = pred_diffs_x * mask.unsqueeze(-1)
        pred_diffs_y = pred_diffs_y * mask.unsqueeze(-1)

        theoretical_diffs_x = y_hat[i,:,0] - y_hat[i-j,:,0]
        theoretical_diffs_y = y_hat[i,:,1] - y_hat[i-j,:,1]

        # Scale the loss so that the latter time steps dont have a larger loss than the earlier time steps
        mask_dim2 = mask.shape[1]
        mask_loss_scale = mask_dim2/mask.sum(dim=1).unsqueeze(1)
        pos_loss_x = torch.mean(((pred_diffs_x-theoretical_diffs_x)*mask_loss_scale.unsqueeze(-1))**2)
        pos_loss_y = torch.mean(((pred_diffs_y-theoretical_diffs_y)*mask_loss_scale.unsqueeze(-1))**2)
        # pos_loss_x = self.loss_func(pred_diffs_x,theoretical_diffs_x)
        # pos_loss_y = self.loss_func(pred_diffs_y,theoretical_diffs_y)

        loss = activity_L2 + pos_loss_x + pos_loss_y
        # print(loss)
        self.losses.append(loss.item())
        return loss
    
    def train_gradual_manual(self,input):
        # Input shape: [Epochs,data/labels,batchsize,tsteps,x/y]
        for i in tqdm(range(len(input))):
            data = input[i][0]
            labels = input[i][1]
            loss = self.train_step(data.to(device),labels.to(device))