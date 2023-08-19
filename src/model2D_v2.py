from sys import stdout

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, functional
import torch.optim as optim

# Liquid Neural Networks
from ncps.wirings import AutoNCP, FullyConnected
from ncps.torch import LTC, LTCCell, CfCCell, WiredCfCCell

import pytorch_lightning as pl
from Sophia import SophiaG

import ratinabox #IMPORT 
from ratinabox.Environment import Environment
from ratinabox.Agent import Agent

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

class RNN_base(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,lr=0.0002,act_decay=0.0,weight_decay=0.01,activation=True,clip_grad=True,optimizer="Sophia"):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.act_decay = act_decay
        self.weight_decay = weight_decay
        self._lr = lr
        self.clip_grad = clip_grad
        self.optimizer_name = optimizer

        # Define activation functions
        self.activation = activation
        if self.activation:
            self.act = torch.nn.ReLU()
        else:
            self.act = torch.nn.Identity()

        self.loss_func = torch.nn.MSELoss()
        self.if_scheduler = False

        self.total_losses = []
        self.task_losses = []

    def train_step(self, x, y_hat):
        self.optimizer.zero_grad(set_to_none=True)
        loss = self.loss_fn(x, y_hat)
        loss.backward()
        if self.clip_grad:
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()
    
    @property
    def lr(self):
        return self._lr
    
    @lr.setter
    def lr(self, lr):
        self._lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def activate_scheduler(self,step_size=100,gamma=0.9):
        self.if_scheduler = True
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def update_optimizer(self):
        if self.optimizer_name == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer_name == "Sophia":
            self.optimizer = SophiaG(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def activate_irnn(self,weights,biases):
        self.irnn = True
        for weight in weights:
            torch.nn.init.eye_(weight)
        for bias in biases:
            torch.nn.init.constant_(bias, 0)

    def plot_losses(self,average=None):
        task_losses = np.array(self.task_losses)
        total_losses = np.array(self.total_losses)
        if self.act_decay == 0.0:
            losses = task_losses
            if average == None:
                plt.plot(losses[10:],label="Task Loss")
            else:
                if len(losses)%average != 0:
                    losses = losses[:-(len(losses)%average)]
                    print("Losses array was not a multiple of average. Truncated to",len(losses))
                loss_data_avgd = losses.reshape(average,-1).mean(axis=1)
                plt.plot(loss_data_avgd[3:],label="Task Loss")
        else:
            losses = total_losses
            if average == None:
                plt.plot(losses[10:],label="Total Loss")
                plt.plot(task_losses[10:],label="Task Loss")
            else:
                if len(losses)%average != 0:
                    losses = losses[:-(len(losses)%average)]
                    task_losses = task_losses[:-(len(task_losses)%average)]
                    print("Losses array was not a multiple of average. Truncated to",len(losses))
                loss_data_avgd = losses.reshape(average,-1).mean(axis=1)
                task_loss_data_avgd = task_losses.reshape(average,-1).mean(axis=1)
                plt.plot(loss_data_avgd[3:],label="Total Loss")
                plt.plot(task_loss_data_avgd[3:],label="Task Loss")
        # Lim to remove the first few spikes
        plt.ylim(0,min(loss_data_avgd.max()*1.1,4))
        plt.title("MSE Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        return total_losses, task_losses
    
    def train_gradual_manual(self,input):
        # Input shape: [Epochs,data/labels,batchsize,tsteps,x/y]
        t = tqdm(range(len(input)), desc="Loss", leave=True)
        for i in t:
            data = input[i][0]
            labels = input[i][1]
            loss = self.train_step(data.to(device),labels.to(device))
            t.set_description(f"Total/Task loss: {self.total_losses[-1]:.5f}/{self.task_losses[-1]:.5f}", refresh=True)
            if self.if_scheduler:
                self.scheduler.step()

class RNN_2D(RNN_base):
    def __init__(self,input_size,hidden_size,output_size=2,lr=0.0002,if_low=False,act_decay=0.0,weight_decay=0.01,act_decay_to_one=False,h_bias=False,input_bias=True,rest_bias=True,irnn=True,activation=True,clip_grad=True,optimizer="Sophia"):
        super().__init__(input_size,hidden_size,output_size,lr,act_decay,weight_decay,activation,clip_grad,optimizer)
        self.act_decay_to_one = act_decay_to_one
        self.if_low = if_low
        if self.if_low:
            self.output_size = 4

        self.input = nn.Linear(self.input_size, self.hidden_size, bias=input_bias)
        self.inputx = nn.Linear(1, self.hidden_size, bias=input_bias)
        self.inputy = nn.Linear(1, self.hidden_size, bias=input_bias)
        self.start_encoder = nn.Linear(self.input_size, self.hidden_size, bias=rest_bias)
        self.hidden = nn.Linear(self.hidden_size, self.hidden_size, bias=h_bias)
        self.output = nn.Linear(self.hidden_size, self.output_size, bias=rest_bias)

        if irnn:
            if input_bias and h_bias:
                self.activate_irnn([self.hidden.weight],[self.input.bias,self.hidden.bias])
            elif input_bias:
                self.activate_irnn([self.hidden.weight],[self.input.bias])
            elif h_bias:
                self.activate_irnn([self.hidden.weight],[self.hidden.bias])
            else:
                self.activate_irnn([self.hidden.weight],[])

        self.update_optimizer()

    def forward(self, x, raw=False,inference=False):
            self.batch_size = x.size(0)
            # Encoder for initial hidden state
            h = self.start_encoder(x[:,0,:,:].squeeze(-1))
            self.time_steps = x.size(1) - 1 # Minus one because first value is initial position
            # time_steps+1 because we want to include the initial hidden state
            self.hts = torch.zeros(self.time_steps+1, self.batch_size, self.hidden_size)
            self.hts[0] = h
            # Main RNN loop
            if not inference:
                for t in range(self.time_steps):
                    h = self.act(self.hidden(h) + self.input(x[:,t+1,:,0]))
                    self.hts[t+1] = h
            else:
                with torch.no_grad():
                    for t in range(self.time_steps):
                        h = self.act(self.hidden(h) + self.input(x[:,t+1,:,0]))
                        self.hts[t+1] = h
            if not raw:
                return self.output(self.hts)
            return self.hts

    def loss_fn(self, x, y_hat):
        if self.if_low:
            y_hat = sincos_from_2D(y_hat)
        y = self(x,raw=True)[1:,:,:]
        # Activity loss
        if self.act_decay_to_one:
            activity_L2 = self.act_decay*((torch.norm(y,dim=-1)-1)**2).sum()
        else:
            activity_L2 = self.act_decay/(self.time_steps*self.hidden_size*self.batch_size)*(y**2).sum()
        y = self.output(y)
        y_hat = y_hat.transpose(0,1)
        task_loss = self.loss_func(y,y_hat)
        loss = task_loss + activity_L2
        self.total_losses.append(loss.item())
        self.task_losses.append(task_loss.item())
        return loss
    

class RNN_arccos_2x1D(RNN_2D):
    def __init__(self,input_size,hidden_size,output_size=2,lr=0.0002,if_low=False,act_decay=0.0,weight_decay=0.01,act_decay_to_one=False,h_bias=False,input_bias=True,rest_bias=True,irnn=True,activation=True,clip_grad=True,optimizer="Sophia"):
        super().__init__(input_size,hidden_size,output_size,lr,if_low,act_decay,weight_decay,act_decay_to_one,h_bias,input_bias,rest_bias,irnn,activation,clip_grad,optimizer)
        assert hidden_size%2 == 0, "Hidden size must be even"

    def loss_fn(self, x, y_hat):
        y = self(x,raw=True)
        # Activity regularization L2
        if not self.act_decay_to_one:
            activity_L2 = self.act_decay/(self.time_steps*self.hidden_size*self.batch_size)*(y**2).sum()

        # Concatenate the start pos to y_hat to make it the same size as y
        y_hat = torch.cat((x[:,0,:,:].squeeze().unsqueeze(1),y_hat),dim=1)
        # Permute y_hat to make it the same shape as y
        y_hat = torch.permute(y_hat,(1,0,2))

        # Split y into x and y components
        h_x = y[:,:,:self.hidden_size//2]
        h_y = y[:,:,self.hidden_size//2:]

        if self.act_decay_to_one:
            activity_L2_y = self.act_decay*((torch.norm(h_y,dim=-1)-1)**2).sum()
            activity_L2_x = self.act_decay*((torch.norm(h_x,dim=-1)-1)**2).sum()
            activity_L2 = activity_L2_y + activity_L2_x

        # Main angle loss loop, checks difference in angles for multiple time steps back in time
        i = torch.arange(1, self.time_steps).unsqueeze(1)
        # Check for 40% of the time steps back in time, to avoid angles beeing too large (above pi) for acos so that they become ambiguous (acos pi-0.1 = acos pi+0.1)
        j = torch.arange(1, max(1,self.time_steps//2-int(self.time_steps*0.1))).unsqueeze(0)
        mask = i >= j
        j = j * mask
        # i = i * mask
        
        ### For x
        normalizer_x = 1 / (torch.norm(h_x[i], dim=-1) * torch.norm(h_x[i-j], dim=-1))
        # Cant clamp between -1 and 1 because it will cause NaNs in training
        angle_test_x = torch.abs(torch.acos(torch.clamp(torch.sum(h_x[i]*h_x[i-j], dim=-1) * normalizer_x, -0.99999, 0.99999)))
        # Make the angles that are not supposed to be checked 0
        angle_test_x = angle_test_x * mask.unsqueeze(-1)
        # Must use torch.abs because the angle can be negative, but the angle_test_x only returns positive angles
        angle_theoretical_x = torch.abs(y_hat[i,:,0]-y_hat[i-j,:,0])
        # Make 0 and 2pi the same angle

        # Scale the loss so that the latter time steps dont have a larger loss than the earlier time steps
        mask_dim2_x = mask.shape[1]
        mask_loss_scale_x = mask_dim2_x/mask.sum(dim=1).unsqueeze(1)
        angle_loss_x = torch.mean(((angle_test_x-angle_theoretical_x)*mask_loss_scale_x.unsqueeze(-1))**2)

        ### For y
        normalizer_y = 1 / (torch.norm(h_y[i], dim=-1) * torch.norm(h_y[i-j], dim=-1))
        angle_test_y = torch.abs(torch.acos(torch.clamp(torch.sum(h_y[i]*h_y[i-j], dim=-1) * normalizer_y, -0.99999, 0.99999)))
        angle_test_y = angle_test_y * mask.unsqueeze(-1)
        angle_theoretical_y = torch.abs(y_hat[i,:,1]-y_hat[i-j,:,1])

        mask_dim2_y = mask.shape[1]
        mask_loss_scale_y = mask_dim2_y/mask.sum(dim=1).unsqueeze(1)
        angle_loss_y = torch.mean(((angle_test_y-angle_theoretical_y)*mask_loss_scale_y.unsqueeze(-1))**2)

        task_loss = angle_loss_x + angle_loss_y
        loss = task_loss + activity_L2
        self.total_losses.append(loss.item())
        self.task_losses.append(task_loss.item())
        return loss
    
class RNN_arccos_3x1D(RNN_2D):
    def __init__(self,input_size,hidden_size,output_size=2,lr=0.0002,if_low=False,act_decay=0.0,weight_decay=0.01,act_decay_to_one=False,h_bias=False,input_bias=True,rest_bias=True,irnn=True,activation=True,clip_grad=True,optimizer="Sophia"):
        super().__init__(input_size,hidden_size,output_size,lr,if_low,act_decay,weight_decay,act_decay_to_one,h_bias,input_bias,rest_bias,irnn,activation,clip_grad,optimizer)
        assert hidden_size%3 == 0, "Hidden size must be divisible by 3"

    def loss_fn(self, x, y_hat):
        y = self(x)
        # Activity regularization L2
        if not self.act_decay_to_one:
            activity_L2 = self.act_decay/(self.time_steps*self.hidden_size*self.batch_size)*(y**2).sum()
        
        y0_hat = convert_23D_single(x[:,0,:,:])
        # Concatenate the start pos to y_hat to make it the same size as y
        y_hat = torch.cat((y0_hat.squeeze().unsqueeze(1),y_hat),dim=1)
        # Permute y_hat to make it the same shape as y
        y_hat = torch.permute(y_hat,(1,0,2))

        h_x = y[:,:,:self.hidden_size//3]
        h_y = y[:,:,self.hidden_size//3:2*self.hidden_size//3]
        h_z = y[:,:,2*self.hidden_size//3:]

        if self.act_decay_to_one:
            activity_L2_x = self.act_decay*((torch.norm(h_x,dim=-1)-1)**2).sum()
            activity_L2_y = self.act_decay*((torch.norm(h_y,dim=-1)-1)**2).sum()
            activity_L2_z = self.act_decay*((torch.norm(h_z,dim=-1)-1)**2).sum()
            activity_L2 = activity_L2_x + activity_L2_y + activity_L2_z
        
        # Main angle loss loop, checks difference in angles for multiple time steps back in time
        i = torch.arange(1, self.time_steps).unsqueeze(1)
        # Check for 40% of the time steps back in time, to avoid angles beeing too large (above pi) for acos so that they become ambiguous (acos pi-0.1 = acos pi+0.1)
        j = torch.arange(1, max(1,self.time_steps//2-int(self.time_steps*0.1))).unsqueeze(0)
        mask = i >= j
        j = j * mask
        # i = i * mask
        
        ### For x
        normalizer_x = 1 / (torch.norm(h_x[i], dim=-1) * torch.norm(h_x[i-j], dim=-1))
        # Cant clamp between -1 and 1 because it will cause NaNs in training
        angle_test_x = torch.abs(torch.acos(torch.clamp(torch.sum(h_x[i]*h_x[i-j], dim=-1) * normalizer_x, -0.99999, 0.99999)))
        # Make the angles that are not supposed to be checked 0
        angle_test_x = angle_test_x * mask.unsqueeze(-1)
        # Must use torch.abs because the angle can be negative, but the angle_test_x only returns positive angles
        angle_theoretical_x = torch.abs(y_hat[i,:,0]-y_hat[i-j,:,0])

        # Scale the loss so that the latter time steps dont have a larger loss than the earlier time steps
        mask_dim2_x = mask.shape[1]
        mask_loss_scale_x = mask_dim2_x/mask.sum(dim=1).unsqueeze(1)
        angle_loss_x = torch.mean(((angle_test_x-angle_theoretical_x)*mask_loss_scale_x.unsqueeze(-1))**2)
        # angle_loss = torch.mean(((angle_test-angle_theoretical))**2)

        ### For y
        normalizer_y = 1 / (torch.norm(h_y[i], dim=-1) * torch.norm(h_y[i-j], dim=-1))
        angle_test_y = torch.abs(torch.acos(torch.clamp(torch.sum(h_y[i]*h_y[i-j], dim=-1) * normalizer_y, -0.99999, 0.99999)))
        angle_test_y = angle_test_y * mask.unsqueeze(-1)
        angle_theoretical_y = torch.abs(y_hat[i,:,1]-y_hat[i-j,:,1])
        mask_dim2_y = mask.shape[1]
        mask_loss_scale_y = mask_dim2_y/mask.sum(dim=1).unsqueeze(1)
        angle_loss_y = torch.mean(((angle_test_y-angle_theoretical_y)*mask_loss_scale_y.unsqueeze(-1))**2)

        ### For z (120 degree)
        normalizer_z = 1 / (torch.norm(h_z[i], dim=-1) * torch.norm(h_z[i-j], dim=-1))
        angle_test_z = torch.abs(torch.acos(torch.clamp(torch.sum(h_z[i]*h_z[i-j], dim=-1) * normalizer_z, -0.99999, 0.99999)))
        angle_test_z = angle_test_z * mask.unsqueeze(-1)
        angle_theoretical_z = torch.abs(y_hat[i,:,2]-y_hat[i-j,:,2])

        mask_dim2_z = mask.shape[1]
        mask_loss_scale_z = mask_dim2_z/mask.sum(dim=1).unsqueeze(1)
        angle_loss_z = torch.mean(((angle_test_z-angle_theoretical_z)*mask_loss_scale_z.unsqueeze(-1))**2)

        task_loss = angle_loss_x + angle_loss_y + angle_loss_z
        loss = task_loss + activity_L2
        self.total_losses.append(loss.item())
        self.task_losses.append(task_loss.item())
        return loss

    def train_gradual_manual(self,input):
        # Input shape: [Epochs,data/labels,batchsize,tsteps,x/y]
        t = tqdm(range(len(input)), desc="Loss", leave=True)
        for i in t:
            data = input[i][0]
            labels = input[i][1]
            _, labels = convert_23D(data,labels)
            loss = self.train_step(data.to(device),labels.to(device))
            t.set_description(f"Total/Task loss: {self.total_losses[-1]:.5f}/{self.task_losses[-1]:.5f}", refresh=True)
            if self.if_scheduler:
                self.scheduler.step()

class CwRNN(RNN_2D):
    def __init__(self, input_size, hidden_size, n_modules,if_low=False,bias=True, lr=0.0002, act_decay=1.0,weight_decay=0.01):
        super().__init__(input_size, hidden_size, lr, act_decay, weight_decay, if_low=if_low)

        self.bias = bias
        self.rnn_cell = nn.RNNCell(input_size, hidden_size,nonlinearity="relu",bias=self.bias)

        assert hidden_size % n_modules == 0
        self.n_modules = n_modules
        self.module_size = hidden_size // n_modules
        self.module_period = [2 ** t for t in range(n_modules)]

        self.update_optimizer()

    def step(self, x, hidden, t):
        """Only update block-rows that correspond to the executed modules."""
        hidden_out = torch.zeros_like(hidden).to(device)
        for i in range(self.n_modules):
            start_row_idx = i * self.module_size
            end_row_idx = (i + 1) * self.module_size
            # check if execute current module
            if t % self.module_period[i] == 0:
                xi = torch.mm(x,
                    self.rnn_cell.weight_ih[
                        start_row_idx:end_row_idx].transpose(0, 1))
                if self.bias:
                    xi = torch.add(xi,
                        self.rnn_cell.bias_ih[start_row_idx:end_row_idx])
                # upper triangular matrix mask
                xh = torch.mm(hidden[:, start_row_idx:],
                    self.rnn_cell.weight_hh[
                    start_row_idx:end_row_idx, start_row_idx:].transpose(0, 1))
                if self.bias:
                    xh = torch.add(xh,
                        self.rnn_cell.bias_hh[start_row_idx:end_row_idx])
                xx = torch.tanh(torch.add(xi, xh))
                hidden_out[:, start_row_idx:end_row_idx] += xx
            else:
                hidden_out[:, start_row_idx:end_row_idx] += \
                    hidden[:, start_row_idx:end_row_idx]
        return hidden_out

    def forward(self, x, raw=False,inference=False):
        b, t, _ = x[:,1:,:,:].squeeze(-1).shape
        # hidden = torch.zeros(b, self.hidden_size).to(device)  # default to zeros
        hidden = self.start_encoder(x[:,0,:,:].squeeze(-1)).to(device)
        y_out = []
        self.hts = torch.zeros(t+1, b, self.hidden_size).to(device)
        self.hts[0] = hidden
        for i in range(t):
            hx = self.step(x[:, i+1,:,:].squeeze(-1), hidden, i)  # (batch_size, hidden_size)
            hidden = hx
            self.hts[i+1] = hx
            hx = self.output(hx)  # (batch_size, output_size)
            y_out.append(hx)

        # output shape (batch_size, seq_len, input_size)
        self.time_steps = t
        self.batch_size = b
        if not inference:
            if not raw:
                return torch.stack(y_out, dim=0).permute(1, 0, 2)
            return self.hts
        else:
            with torch.no_grad():
                if not raw:
                    return torch.stack(y_out, dim=0).permute(1, 0, 2)
                return self.hts
            

class LSTM_solver(RNN_2D):
    def __init__(self,input_size,hidden_size,output_size=2,lr=0.0002,if_low=False,act_decay=1.0,weight_decay=0.01,irnn=True,activation=True):
        super().__init__(input_size,hidden_size,output_size,lr=lr,if_low=if_low,act_decay=act_decay,weight_decay=weight_decay,irnn=irnn,activation=activation)
        self.lstm = nn.LSTMCell(self.input_size,self.hidden_size,bias=False)
        self.c_encoder = nn.Linear(self.input_size,self.hidden_size,bias=False)

        self.update_optimizer()

    def forward(self, x, raw=False,inference=False):
        x = x.squeeze(-1)
        b, t, _ = x[:,1:,:].shape
        # hidden = torch.zeros(b, self.hidden_size)  # default to zeros
        hidden = self.start_encoder(x[:,0,:])
        cell = self.c_encoder(x[:,0,:])
        y_out = []
        self.hts = torch.zeros(t+1, b, self.hidden_size)
        self.cts = torch.zeros(t+1, b, self.hidden_size)
        self.hts[0] = hidden
        self.cts[0] = cell
        for i in range(t):
            hx, cx = self.lstm(x[:, i+1,:], (hidden,cell))  # (batch_size, hidden_size)
            hidden = hx
            cell = cx
            self.hts[i+1] = hx
            self.cts[i+1] = cx
            hx = self.output(hx)  # (batch_size, output_size)
            y_out.append(hx)

        # output shape (batch_size, seq_len, input_size)
        self.time_steps = t
        self.batch_size = b
        if not inference:
            if not raw:
                return torch.stack(y_out, dim=0)
            return self.hts.permute(1,0,2)
        else:
            with torch.no_grad():
                if not raw:
                    return torch.stack(y_out, dim=0)
                return self.hts.permute(1,0,2)
    

class CfC_solver(RNN_2D):
    def __init__(self,input_size,hidden_size,output_size=2,lr=0.0002,if_low=False,act_decay=1.0,weight_decay=0.01,if_lstm=True,if_wiring=True):
        super().__init__(input_size,hidden_size,output_size,lr=lr,if_low=if_low,act_decay=act_decay,weight_decay=weight_decay)
        self.if_wiring = if_wiring
        if self.if_wiring:
            self.wiring = AutoNCP(self.hidden_size, self.output_size)
        else:
            self.wiring = FullyConnected(self.hidden_size, output_dim=self.output_size)
        self.if_lstm = if_lstm
        if self.if_lstm:
            self.lstm_cell = nn.LSTMCell(self.input_size,self.hidden_size,bias=False)
            self.c_encoder = nn.Linear(self.input_size,self.hidden_size,bias=False)
        self.cfc = WiredCfCCell(self.input_size,self.wiring)

        self.update_optimizer()

    def forward(self, x, raw=False, inference=False):
        x = x.squeeze(-1)
        b, t, _ = x[:,1:,:].shape
        hidden = self.start_encoder(x[:,0,:])
        cell = self.c_encoder(x[:,0,:])
        y_out = []
        ts = 1.0
        if not inference:
            self.hts = torch.zeros(t+1, b, self.hidden_size)
            self.hts[0] = hidden
            for i in range(t):
                if self.if_lstm:
                    hx, cx = self.lstm_cell(x[:, i+1,:], (hidden,cell))
                y, hx = self.cfc(x[:, i+1,:], hidden, timespans=ts)  # (batch_size, hidden_size)
                hidden = hx
                cell = cx
                self.hts[i+1] = hx
                y_out.append(y) # (batch_size, output_size)    
        else:
            with torch.no_grad():
                self.hts = np.zeros((t+1, b, self.hidden_size))
                self.hts[0] = hidden
                for i in range(t):
                    if self.if_lstm:
                        hx, cx = self.lstm_cell(x[:, i+1,:], (hidden,cell))
                    y, hx = self.cfc(x[:, i+1,:], hidden,timespans=ts)
                    hidden = hx
                    cell = cx
                    self.hts[i+1] = hx
                    y_out.append(y) # (batch_size, output_size)
            
        self.time_steps = t
        self.batch_size = b
        # output shape (batch_size, seq_len, input_size)
        if not inference:
            if not raw:
                return torch.stack(y_out, dim=0)
            return self.hts.permute(1,0,2)
        else:
            with torch.no_grad():
                if not raw:
                    return torch.stack(y_out, dim=0)
                return self.hts.permute(1,0,2)