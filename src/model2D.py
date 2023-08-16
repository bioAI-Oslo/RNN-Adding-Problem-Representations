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

# LightningModule for training a RNNSequence module
class SequenceLearner_xy(pl.LightningModule):
    def __init__(self, model, lr=0.005):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model.forward(x)
        y_hat = y_hat.view_as(y)
        loss = nn.MSELoss()(y_hat, y)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return SophiaG(self.model.parameters(), lr=self.lr)
        # return torch.optim.Adam(self.model.parameters(), lr=self.lr)


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
                torch.nn.init.constant_(self.inputx.bias, 0)
                torch.nn.init.constant_(self.inputx.bias, 0)

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

    def forward(self, x, raw=False,inference=False):
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
        if not inference:
            if not raw:
                return self.output(self.hts)
            return self.hts
        else:
            if not raw:
                return self.output(self.hts).cpu().detach().numpy()
            return self.hts.cpu().detach().numpy()

    def loss_fn(self, x, y_hat):
        y = self(x,raw=True)[1:,:,:]
        # Activity loss
        activity_L2 = self.act_decay*((torch.norm(y,dim=-1)-1)**2).sum()
        y = self.output(y)
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
        # Lim to remove the first few spikes
        plt.ylim(0,min(loss_data_avgd.max()*1.1,4))
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
        t = tqdm(range(len(input)), desc="Loss", leave=True)
        for i in t:
            data = input[i][0]
            labels = input[i][1]
            labels = sincos_from_2D(labels)
            loss = self.train_step(data.to(device),labels.to(device))
            t.set_description(f"Loss: {loss:.5f}", refresh=True)

            
            

class RNN_circular_2D_xy_Low_randomstart(RNN_circular_2D_xy_Low):
    def __init__(self,input_size,hidden_size,lr=0.0005,act_decay=0.0,weight_decay=0.01,noise=0.1,irnn=True,outputnn=True,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=2):
        super().__init__(input_size,hidden_size,lr=lr,act_decay=act_decay,weight_decay=weight_decay,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space)
        self.start_encoder = torch.nn.Linear(self.nav_space,self.hidden_size,bias=False)
        self.noise = noise

        self.optimizer = SophiaG(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x, raw=False,inference=False):
            # Make h0 trainable
            batch_size_forward = x.size(0)
            # Encoder for initial hidden state
            # h = torch.zeros((batch_size_forward,self.hidden_size)) # Gives constant initial hidden state
            h = self.start_encoder(x[:,0,:,:].squeeze(-1))
            self.time_steps = x.size(1) - 1 # Minus one because first value is initial position
            # time_steps+1 because we want to include the initial hidden state
            self.hts = torch.zeros(self.time_steps+1, batch_size_forward, self.hidden_size)
            self.hts[0] = h
            # Main RNN loop
            for t in range(self.time_steps):
                # print(x.shape)
                h = self.act(self.hidden(h) + self.inputx(x[:,t+1,0,:]) + self.inputy(x[:,t+1,1,:])) + torch.normal(0,self.noise,(batch_size_forward,self.hidden_size)).to(device)
                self.hts[t+1] = h
            if not inference:
                if not raw:
                    return self.output(self.hts)
                return self.hts
            else:
                if not raw:
                    return self.output(self.hts).cpu().detach().numpy()
                return self.hts.cpu().detach().numpy()
    
    def loss_fn(self, x, y_hat):
        y = self(x,raw=True)[1:,:,:]
        # Activity loss
        # activity_L2 = self.act_decay*((torch.norm(y,dim=-1)-1)**2).sum()
        activity_L2 = self.act_decay/(self.time_steps*self.hidden_size*self.batch_size)*(y**2).sum()
        y = self.output(y)
        y_hat = y_hat.transpose(0,1)
        loss_sin_x = self.loss_func(y[:,:,0],y_hat[:,:,0])
        loss_cos_x = self.loss_func(y[:,:,1],y_hat[:,:,1])
        loss_sin_y = self.loss_func(y[:,:,2],y_hat[:,:,2])
        loss_cos_y = self.loss_func(y[:,:,3],y_hat[:,:,3])
        loss = loss_sin_x + loss_cos_x + loss_sin_y + loss_cos_y + activity_L2
        self.losses.append(loss.item())
        return loss
    
class RNN_circular_2D_randomstart_trivial_sorcher(RNN_circular_2D_xy_Low_randomstart):
    def __init__(self,input_size,hidden_size,lr=0.0002,act_decay=0.0,weight_decay=0.01,noise=0.05,irnn=True,outputnn=True,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=2):
        super().__init__(input_size,hidden_size,lr=lr,act_decay=act_decay,weight_decay=weight_decay,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space)

        self.noise = noise
        self.output = torch.nn.Linear(self.hidden_size,self.nav_space,bias=bias)
        self.optimizer = SophiaG(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x, raw=False,inference=False):
            # Make h0 trainable
            batch_size_forward = x.size(0)
            # Encoder for initial hidden state
            # h = torch.zeros((batch_size_forward,self.hidden_size)) # Gives constant initial hidden state
            h = self.start_encoder(x[:,0,:,:].squeeze(-1))
            self.time_steps = x.size(1) - 1 # Minus one because first value is initial position
            # time_steps+1 because we want to include the initial hidden state
            self.hts = torch.zeros(self.time_steps+1, batch_size_forward, self.hidden_size)
            self.hts[0] = h
            # Main RNN loop
            for t in range(self.time_steps):
                # print(x.shape)
                h = self.act(self.hidden(h) + self.inputx(x[:,t+1,0,:]) + self.inputy(x[:,t+1,1,:])) + torch.normal(0,self.noise,(batch_size_forward,self.hidden_size)).to(device)
                self.hts[t+1] = h
            if not inference:
                if not raw:
                    return self.output(self.hts)
                return self.hts
            else:
                if not raw:
                    return self.output(self.hts).cpu().detach().numpy()
                return self.hts.cpu().detach().numpy()
    
    def loss_fn(self, x, y_hat):
        y = self(x,raw=True)[1:,:,:]
        # Activity loss
        activity_L2 = self.act_decay/(self.time_steps*self.hidden_size*self.batch_size)*(y**2).sum()
        y = self.output(y)
        y_hat = y_hat.transpose(0,1)
        loss_x = self.loss_func(y[:,:,0],y_hat[:,:,0])
        loss_y = self.loss_func(y[:,:,1],y_hat[:,:,1])
        loss = loss_x + loss_y + activity_L2
        self.losses.append(loss.item())
        return loss
    
    def train(self, epochs=100, loader=None):
        for epoch in tqdm(range(epochs)):
            # data,labels = datagen_circular_pm(self.batch_size,self.base_training_tsteps,sigma=0.05)
            data,labels = smooth_wandering_2D_squarefix_randomstart_hdv(self.batch_size,self.base_training_tsteps)
            loss = self.train_step(data.to(device),labels.to(device))
        return self.losses
    
    def train_gradual(self, epochs=100, loader=None):
        i = 0
        training_steps = 1
        for epoch in tqdm(range(epochs)):
            if i%15 == 0:
                training_steps += 1
            data,labels = smooth_wandering_2D_squarefix_randomstart_hdv(self.batch_size,training_steps)
            loss = self.train_step(data.to(device),labels.to(device))
            i+=1
        print("Last training time steps:",training_steps)
        return self.losses
    
    def train_gradual_manual(self,input):
        # Input shape: [Epochs,data/labels,batchsize,tsteps,x/y]
        t = tqdm(range(len(input)), desc="Loss", leave=True)
        for i in t:
            data = input[i][0]
            labels = input[i][1]
            loss = self.train_step(data.to(device),labels.to(device))
            t.set_description(f"Loss: {loss:.5f}", refresh=True)

    def train_gradual_loader(self,data_loader):
        # Input shape: [Epochs,data/labels,batchsize,tsteps,x/y]
        for (data,labels) in tqdm(data_loader):
            data = data.squeeze(0)
            labels = labels.squeeze(0)
            loss = self.train_step(data.to(device),labels.to(device))

class RNN_circular_1D_to_2D_arccos(RNN_circular_2D_randomstart_trivial_sorcher):
    def __init__(self,input_size,hidden_size,lr=0.0005,act_decay=0.0,weight_decay=0.01,noise=0.05,irnn=True,outputnn=True,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=2):
        super().__init__(input_size,hidden_size,lr=lr,act_decay=act_decay,weight_decay=weight_decay,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space)

    def forward(self, x, raw=False,inference=False):
        # Make h0 trainable
        batch_size_forward = x.size(0)
        # Encoder for initial hidden state
        # h = torch.zeros((batch_size_forward,self.hidden_size)) # Gives constant initial hidden state
        h = self.start_encoder(x[:,0,:,:].squeeze(-1))
        self.time_steps = x.size(1) - 1 # Minus one because first value is initial position
        # time_steps+1 because we want to include the initial hidden state
        self.hts = torch.zeros(self.time_steps+1, batch_size_forward, self.hidden_size)
        self.hts[0] = h
        # Main RNN loop
        for t in range(self.time_steps):
            # print(x.shape)
            h = self.act(self.hidden(h) + self.inputx(x[:,t+1,0,:]) + self.inputy(x[:,t+1,1,:])) + torch.normal(0,self.noise,(batch_size_forward,self.hidden_size)).to(device)
            self.hts[t+1] = h
        if not inference:
            return self.hts
        else:
            return self.hts.cpu().detach().numpy()

    def loss_fn(self, x, y_hat):
        y = self(x)
        # Activity regularization L2
        activity_L2 = self.act_decay/(self.time_steps*self.hidden_size*self.batch_size)*(y**2).sum()
        # activity_L2 = self.act_decay*((torch.norm(y,dim=-1)-1)**2).sum()
        
        # Concatenate the start pos to y_hat to make it the same size as y
        y_hat = torch.cat((x[:,0,:,:].squeeze().unsqueeze(1),y_hat),dim=1)
        # Permute y_hat to make it the same shape as y
        y_hat = torch.permute(y_hat,(1,0,2))

        h_x = y[:,:,:self.hidden_size//2]
        h_y = y[:,:,self.hidden_size//2:]

        # activity_L2_x = self.act_decay*((torch.norm(h_x,dim=-1)-1)**2).sum()
        # activity_L2_y = self.act_decay*((torch.norm(h_y,dim=-1)-1)**2).sum()

        
        # Main angle loss loop, checks difference in angles for multiple time steps back in time
        i = torch.arange(1, self.time_steps).unsqueeze(1)
        # Check for 40% of the time steps back in time, to avoid angles beeing too large (above pi) for acos so that they become ambiguous (acos pi-0.1 = acos pi+0.1)
        j = torch.arange(1, max(1,self.time_steps//2-int(self.time_steps*0.1))).unsqueeze(0)
        # j = torch.arange(1, max(1,self.time_steps)).unsqueeze(0)
        # THIS IS VERY UNCERTAIN, TRY >= AND >
        mask = i >= j
        j = j * mask
        # i = i * mask
        
        ### For x
        normalizer_x = 1 / (torch.norm(h_x[i], dim=-1) * torch.norm(h_x[i-j], dim=-1))
        # Cant clamp between -1 and 1 because it will cause NaNs in training
        angle_test_x = torch.abs(torch.acos(torch.clamp(torch.sum(h_x[i]*h_x[i-j], dim=-1) * normalizer_x, -0.9999999, 0.9999999)))
        # Make the angles that are not supposed to be checked 0
        angle_test_x = angle_test_x * mask.unsqueeze(-1)
        # Must use torch.abs because the angle can be negative, but the angle_test_x only returns positive angles
        angle_theoretical_x = torch.abs(y_hat[i,:,0]-y_hat[i-j,:,0])
        # Make 0 and 2pi the same angle

        # Scale the loss so that the latter time steps dont have a larger loss than the earlier time steps
        mask_dim2_x = mask.shape[1]
        mask_loss_scale_x = mask_dim2_x/mask.sum(dim=1).unsqueeze(1)
        angle_loss_x = torch.mean(((angle_test_x-angle_theoretical_x)*mask_loss_scale_x.unsqueeze(-1))**2)
        # angle_loss = torch.mean(((angle_test-angle_theoretical))**2)

        ### For y
        normalizer_y = 1 / (torch.norm(h_y[i], dim=-1) * torch.norm(h_y[i-j], dim=-1))
        # Cant clamp between -1 and 1 because it will cause NaNs in training
        angle_test_y = torch.abs(torch.acos(torch.clamp(torch.sum(h_y[i]*h_y[i-j], dim=-1) * normalizer_y, -0.9999999, 0.9999999)))
        # Make the angles that are not supposed to be checked 0
        angle_test_y = angle_test_y * mask.unsqueeze(-1)
        # Must use torch.abs because the angle can be negative, but the angle_test_y only returns positive angles
        angle_theoretical_y = torch.abs(y_hat[i,:,1]-y_hat[i-j,:,1])
        # Make 0 and 2pi the same angle
        # angle_theoretical = torch.min(torch.remainder(2*np.pi-(y_hat[i]-y_hat[i-j]),2*np.pi),torch.remainder(2*np.pi-(y_hat[i-j]-y_hat[i]),2*np.pi))

        # Scale the loss so that the latter time steps dont have a larger loss than the earlier time steps
        mask_dim2_y = mask.shape[1]
        mask_loss_scale_y = mask_dim2_y/mask.sum(dim=1).unsqueeze(1)
        angle_loss_y = torch.mean(((angle_test_y-angle_theoretical_y)*mask_loss_scale_y.unsqueeze(-1))**2)

        # Loss to end in the same position as the start
        # circle_end_loss = 0.0001*torch.mean((y[-1]-y[0])**2)

        self.losses.append(angle_loss_x.item() + angle_loss_y.item())
        # self.losses_norm.append(activity_L2.item())
        # self.losses_circle.append(circle_end_loss.item())
        loss = angle_loss_x + angle_loss_y + activity_L2
        return loss
    
class RNN_circular_1D_to_23D_arccos(RNN_circular_1D_to_2D_arccos):
    def __init__(self,input_size,hidden_size,lr=0.0005,act_decay=0.0,weight_decay=0.01,noise=0.05,irnn=True,outputnn=True,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=3):
        super().__init__(input_size,hidden_size,lr=lr,act_decay=act_decay,weight_decay=weight_decay,noise=noise,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space)

    def loss_fn(self, x, y_hat):
        y = self(x)
        # Activity regularization L2
        activity_L2 = self.act_decay/(self.time_steps*self.hidden_size*self.batch_size)*(y**2).sum()
        # activity_L2 = self.act_decay*((torch.norm(y,dim=-1)-1)**2).sum()
        
        # Concatenate the start pos to y_hat to make it the same size as y
        y_hat = torch.cat((x[:,0,:,:].squeeze().unsqueeze(1),y_hat),dim=1)
        # Permute y_hat to make it the same shape as y
        y_hat = torch.permute(y_hat,(1,0,2))

        h_x = y[:,:,:self.hidden_size//3]
        h_y = y[:,:,self.hidden_size//3:2*self.hidden_size//3]
        h_z = y[:,:,2*self.hidden_size//3:]


        # activity_L2_x = self.act_decay*((torch.norm(h_x,dim=-1)-1)**2).sum()
        # activity_L2_y = self.act_decay*((torch.norm(h_y,dim=-1)-1)**2).sum()

        
        # Main angle loss loop, checks difference in angles for multiple time steps back in time
        i = torch.arange(1, self.time_steps).unsqueeze(1)
        # Check for 40% of the time steps back in time, to avoid angles beeing too large (above pi) for acos so that they become ambiguous (acos pi-0.1 = acos pi+0.1)
        j = torch.arange(1, max(1,self.time_steps//2-int(self.time_steps*0.1))).unsqueeze(0)
        # THIS IS VERY UNCERTAIN, TRY >= AND >
        mask = i >= j
        j = j * mask
        # i = i * mask
        
        ### For x
        normalizer_x = 1 / (torch.norm(h_x[i], dim=-1) * torch.norm(h_x[i-j], dim=-1))
        # Cant clamp between -1 and 1 because it will cause NaNs in training
        angle_test_x = torch.abs(torch.acos(torch.clamp(torch.sum(h_x[i]*h_x[i-j], dim=-1) * normalizer_x, -0.9999999, 0.9999999)))
        # Make the angles that are not supposed to be checked 0
        angle_test_x = angle_test_x * mask.unsqueeze(-1)
        # Must use torch.abs because the angle can be negative, but the angle_test_x only returns positive angles
        angle_theoretical_x = torch.abs(y_hat[i,:,0]-y_hat[i-j,:,0])
        # Make 0 and 2pi the same angle

        # Scale the loss so that the latter time steps dont have a larger loss than the earlier time steps
        mask_dim2_x = mask.shape[1]
        mask_loss_scale_x = mask_dim2_x/mask.sum(dim=1).unsqueeze(1)
        angle_loss_x = torch.mean(((angle_test_x-angle_theoretical_x)*mask_loss_scale_x.unsqueeze(-1))**2)
        # angle_loss = torch.mean(((angle_test-angle_theoretical))**2)

        ### For y
        normalizer_y = 1 / (torch.norm(h_y[i], dim=-1) * torch.norm(h_y[i-j], dim=-1))
        # Cant clamp between -1 and 1 because it will cause NaNs in training
        angle_test_y = torch.abs(torch.acos(torch.clamp(torch.sum(h_y[i]*h_y[i-j], dim=-1) * normalizer_y, -0.9999999, 0.9999999)))
        # Make the angles that are not supposed to be checked 0
        angle_test_y = angle_test_y * mask.unsqueeze(-1)
        # Must use torch.abs because the angle can be negative, but the angle_test_y only returns positive angles
        angle_theoretical_y = torch.abs(y_hat[i,:,1]-y_hat[i-j,:,1])
        # Make 0 and 2pi the same angle
        # angle_theoretical = torch.min(torch.remainder(2*np.pi-(y_hat[i]-y_hat[i-j]),2*np.pi),torch.remainder(2*np.pi-(y_hat[i-j]-y_hat[i]),2*np.pi))

        # Scale the loss so that the latter time steps dont have a larger loss than the earlier time steps
        mask_dim2_y = mask.shape[1]
        mask_loss_scale_y = mask_dim2_y/mask.sum(dim=1).unsqueeze(1)
        angle_loss_y = torch.mean(((angle_test_y-angle_theoretical_y)*mask_loss_scale_y.unsqueeze(-1))**2)

        ### For z (120 degree)
        normalizer_z = 1 / (torch.norm(h_z[i], dim=-1) * torch.norm(h_z[i-j], dim=-1))
        # Cant clamp between -1 and 1 because it will cause NaNs in training
        angle_test_z = torch.abs(torch.acos(torch.clamp(torch.sum(h_z[i]*h_z[i-j], dim=-1) * normalizer_z, -0.9999999, 0.9999999)))
        # Make the angles that are not supposed to be checked 0
        angle_test_z = angle_test_z * mask.unsqueeze(-1)
        # Must use torch.abs because the angle can be negative, but the angle_test_z onlz returns positive angles
        angle_theoretical_z = torch.abs(y_hat[i,:,2]-y_hat[i-j,:,2])
        # Make 0 and 2pi the same angle
        # angle_theoretical = torch.min(torch.remainder(2*np.pi-(y_hat[i]-y_hat[i-j]),2*np.pi),torch.remainder(2*np.pi-(y_hat[i-j]-y_hat[i]),2*np.pi))

        # Scale the loss so that the latter time steps dont have a larger loss than the earlier time steps
        mask_dim2_z = mask.shape[1]
        mask_loss_scale_z = mask_dim2_z/mask.sum(dim=1).unsqueeze(1)
        angle_loss_z = torch.mean(((angle_test_z-angle_theoretical_z)*mask_loss_scale_z.unsqueeze(-1))**2)

        # Loss to end in the same position as the start
        # circle_end_loss = 0.0001*torch.mean((y[-1]-y[0])**2)

        self.losses.append(angle_loss_x.item() + angle_loss_y.item())
        # self.losses_norm.append(activity_L2.item())
        # self.losses_circle.append(circle_end_loss.item())
        loss = angle_loss_x + angle_loss_y + angle_loss_z + activity_L2
        return loss


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
        # # Check for 40% of the time steps back in time, to reduce the number of comparisons (short term memory)
        # j = torch.arange(1, max(self.time_steps//2-int(self.time_steps*0.1),1)).unsqueeze(0)
        j = torch.arange(1, max(1,self.time_steps-1)).unsqueeze(0)
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

        # start_loss = self.loss_func(y[0,:,:],y_hat[0,:,:])
        loss = activity_L2 + pos_loss_x + pos_loss_y #+ 0.0001*start_loss
        # print(loss)
        self.losses.append(loss.item())
        return loss
    
    def train_gradual_manual(self,input):
        # Input shape: [Epochs,data/labels,batchsize,tsteps,x/y]
        for i in tqdm(range(len(input))):
            data = input[i][0]
            labels = input[i][1]
            loss = self.train_step(data.to(device),labels.to(device))


class RNN_circular_2D_xy_relative_randomstart(RNN_circular_2D_xy_relative):
    def __init__(self, input_size, hidden_size, lr=0.0005, act_decay=0, weight_decay=0.01, irnn=True, outputnn=True, bias=False, Wx_normalize=False, activation=True, batch_size=64, nav_space=2):
        super().__init__(input_size, hidden_size, lr, act_decay, weight_decay, irnn, outputnn, bias, Wx_normalize, activation, batch_size, nav_space)
        self.start_encoder = torch.nn.Linear(self.nav_space,self.hidden_size,bias=False)

        self.optimizer = SophiaG(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def forward(self, x, raw=False,inference=False):
            # Make h0 trainable
            batch_size_forward = x.size(0)
            # Encoder for initial hidden state
            # h = torch.zeros((batch_size_forward,self.hidden_size)) # Gives constant initial hidden state
            h = self.start_encoder(x[:,0,:,:].squeeze(-1))
            self.time_steps = x.size(1) - 1 # Minus one because first value is initial position
            # time_steps+1 because we want to include the initial hidden state
            self.hts = torch.zeros(self.time_steps+1, batch_size_forward, self.hidden_size)
            self.hts[0] = h
            # Main RNN loop
            for t in range(self.time_steps):
                # print(x.shape)
                h = self.act(self.hidden(h) + self.inputx(x[:,t+1,0,:]) + self.inputy(x[:,t+1,1,:]))
                self.hts[t+1] = h
            if not inference:
                if not raw:
                    return self.output(self.hts)
                return self.hts
            else:
                if not raw:
                    return self.output(self.hts).cpu().detach().numpy()
                return self.hts.cpu().detach().numpy()
    

class CwRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_modules,bias=False, lr=0.0002, act_decay=1.0,weight_decay=0.01):
        super(CwRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn_cell = nn.RNNCell(input_size, hidden_size,nonlinearity="relu",bias=bias)
        self.fc = nn.Linear(hidden_size, input_size)
        self.start_encoder = nn.Linear(input_size,self.hidden_size,bias=False)

        # Initialize IRNN
        nn.init.eye_(self.rnn_cell.weight_hh)

        assert hidden_size % n_modules == 0
        self.n_modules = n_modules
        self.module_size = hidden_size // n_modules
        self.module_period = [2 ** t for t in range(n_modules)]

        self.losses = []

        self.act_decay = act_decay

        self.loss_func = torch.nn.MSELoss()

        # SophiaG optimizer
        self.optimizer = SophiaG(self.parameters(), lr=lr, weight_decay=weight_decay)

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
                # xi = torch.add(xi,
                    # self.rnn_cell.bias_ih[start_row_idx:end_row_idx])

                # upper triangular matrix mask
                xh = torch.mm(hidden[:, start_row_idx:],
                    self.rnn_cell.weight_hh[
                    start_row_idx:end_row_idx, start_row_idx:].transpose(0, 1))

                # xh = torch.add(xh,
                    # self.rnn_cell.bias_hh[start_row_idx:end_row_idx])
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
        x_out = []
        self.hts = torch.zeros(t+1, b, self.hidden_size).to(device)
        self.hts[0] = hidden
        for i in range(t):
            hx = self.step(x[:, i+1,:,:].squeeze(-1), hidden, i)  # (batch_size, hidden_size)
            hidden = hx
            self.hts[i+1] = hx
            hx = self.fc(hx)  # (batch_size, output_size)
            x_out.append(hx)

        # output shape (batch_size, seq_len, input_size)
        self.time_steps = t
        self.batch_size = b
        if not inference:
            if not raw:
                return torch.stack(x_out, dim=0).permute(1, 0, 2)
            return self.hts
        else:
            if not raw:
                return torch.stack(x_out, dim=0).permute(1, 0, 2).cpu().detach().numpy()
            return self.hts.cpu().detach().numpy()
    
    def loss_fn(self, x, y_hat):
        hts = self(x,raw=True)
        # Activity loss
        activity_L2 = self.act_decay/(self.time_steps*self.hidden_size*self.batch_size)*(hts**2).sum()
        y = self(x,raw=False)
        # y_hat = y_hat.transpose(0,1)
        loss_x = self.loss_func(y[:,:,0],y_hat[:,:,0])
        loss_y = self.loss_func(y[:,:,1],y_hat[:,:,1])
        loss = loss_x + loss_y + activity_L2
        self.losses.append(loss.item())
        return loss

    def train_step(self, x, y_hat):
        self.optimizer.zero_grad(set_to_none=True)
        # Backward hook to clip gradients
        loss = self.loss_fn(x, y_hat)
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.parameters(), 2.0) # Clip gradients as in paper Low et al.
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
        average_total = losses.mean()
        plt.ylim(0,min(loss_data_avgd.max()*1.1,4))
        plt.title("MSE Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()
    
    def train_gradual_manual(self,input):
        # Input shape: [Epochs,data/labels,batchsize,tsteps,x/y]
        for i in tqdm(range(len(input))):
            data = input[i][0]
            labels = input[i][1]
            loss = self.train_step(data.to(device),labels.to(device))


class CwRNN_low(CwRNN):
    def __init__(self, input_size, hidden_size,output_size=4, n_modules=3, lr=0.0002, weight_decay=0.01, bias=False, act_decay=1.0):
        super(CwRNN_low, self).__init__(input_size, hidden_size, n_modules, lr, weight_decay, bias, act_decay)
        self.fc = nn.Linear(self.hidden_size, output_size)
        self.loss_func = nn.MSELoss()
        self.optimizer = SophiaG(self.parameters(), lr=lr, weight_decay=weight_decay)

    def loss_fn(self, x, y_hat):
        hts = self(x,raw=True)
        # Activity loss
        activity_L2 = self.act_decay/(self.time_steps*self.hidden_size*self.batch_size)*(hts**2).sum()
        y = self(x,raw=False)
        # y_hat = y_hat.transpose(0,1)
        loss_sin_x = self.loss_func(y[:,:,0],y_hat[:,:,0])
        loss_cos_x = self.loss_func(y[:,:,1],y_hat[:,:,1])
        loss_sin_y = self.loss_func(y[:,:,2],y_hat[:,:,2])
        loss_cos_y = self.loss_func(y[:,:,3],y_hat[:,:,3])
        loss = loss_sin_x + loss_cos_x + loss_sin_y + loss_cos_y + activity_L2
        self.losses.append(loss.item())
        return loss
    
    def train_gradual_manual(self,input):
        # Input shape: [Epochs,data/labels,batchsize,tsteps,x/y]
        t = tqdm(range(len(input)), desc="Loss", leave=True)
        for i in t:
            data = input[i][0]
            labels = input[i][1]
            labels = sincos_from_2D(labels)
            loss = self.train_step(data.to(device),labels.to(device))
            t.set_description(f"Loss: {loss:.5f}", refresh=True)

class LSTM_solver(RNN_circular_2D_randomstart_trivial_sorcher):
    def __init__(self,input_size,hidden_size,output_size=2,lr=0.0002,act_decay=1.0,weight_decay=0.01,irnn=True,outputnn=True,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=2):
        super().__init__(input_size,hidden_size,lr=lr,act_decay=act_decay,weight_decay=weight_decay,noise=0,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space)
        self.output_size = output_size
        self.lstm = nn.LSTMCell(self.input_size,self.hidden_size,bias=False)

        self.c_encoder = nn.Linear(self.input_size,self.hidden_size,bias=False)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        self.optimizer = SophiaG(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x, raw=False,inference=False):
        x = x.squeeze(-1)
        b, t, _ = x[:,1:,:].shape
        # hidden = torch.zeros(b, self.hidden_size)  # default to zeros
        hidden = self.start_encoder(x[:,0,:])
        cell = self.c_encoder(x[:,0,:])
        x_out = []
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
            hx = self.fc(hx)  # (batch_size, output_size)
            x_out.append(hx)

        # output shape (batch_size, seq_len, input_size)
        self.time_steps = t
        self.batch_size = b
        if not inference:
            if not raw:
                return torch.stack(x_out, dim=0).permute(1, 0, 2)
            return self.hts
        else:
            if not raw:
                return torch.stack(x_out, dim=0).permute(1, 0, 2).cpu().detach().numpy()
            return self.hts.cpu().detach().numpy()
    
    def loss_fn(self, x, y_hat):
        hts = self(x,raw=True)
        # Activity loss
        activity_L2 = self.act_decay/(self.time_steps*self.hidden_size*self.batch_size)*(hts**2).sum()
        y = self(x,raw=False)
        # y_hat = y_hat.transpose(0,1)
        loss_x = self.loss_func(y[:,:,0],y_hat[:,:,0])
        loss_y = self.loss_func(y[:,:,1],y_hat[:,:,1])
        loss = loss_x + loss_y + activity_L2
        self.losses.append(loss.item())
        return loss
    
class LSTM_solver_Low(LSTM_solver):
    def __init__(self,input_size,hidden_size,output_size=4,lr=0.0002,act_decay=1.0,weight_decay=0.01,irnn=True,outputnn=True,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=2):
        super().__init__(input_size,hidden_size,lr=lr,act_decay=act_decay,weight_decay=weight_decay,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space)
        self.fc = nn.Linear(self.hidden_size, output_size)

        self.optimizer = SophiaG(self.parameters(), lr=lr, weight_decay=weight_decay)

    def loss_fn(self, x, y_hat):
        hts = self(x,raw=True)
        # Activity loss
        activity_L2 = self.act_decay/(self.time_steps*self.hidden_size*self.batch_size)*(hts**2).sum()
        y = self(x,raw=False)
        # y_hat = y_hat.transpose(0,1)
        loss_sin_x = self.loss_func(y[:,:,0],y_hat[:,:,0])
        loss_cos_x = self.loss_func(y[:,:,1],y_hat[:,:,1])
        loss_sin_y = self.loss_func(y[:,:,2],y_hat[:,:,2])
        loss_cos_y = self.loss_func(y[:,:,3],y_hat[:,:,3])
        loss = loss_sin_x + loss_cos_x + loss_sin_y + loss_cos_y + activity_L2
        self.losses.append(loss.item())
        return loss
    
    def train_gradual_manual(self,input):
        # Input shape: [Epochs,data/labels,batchsize,tsteps,x/y]
        t = tqdm(range(len(input)), desc="Loss", leave=True)
        for i in t:
            data = input[i][0]
            labels = input[i][1]
            labels = sincos_from_2D(labels)
            loss = self.train_step(data.to(device),labels.to(device))
            t.set_description(f"Loss: {loss:.5f}", refresh=True)

class LTC_solver(LSTM_solver):
    def __init__(self,input_size,hidden_size,output_size=2,lr=0.0002,act_decay=1.0,weight_decay=0.01,irnn=True,outputnn=True,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=2, if_lstm=False, if_wiring=True):
        super().__init__(input_size,hidden_size,output_size=output_size,lr=lr,act_decay=act_decay,weight_decay=weight_decay,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space)
        self.if_wiring = if_wiring
        if self.if_wiring:
            self.wiring = AutoNCP(self.hidden_size, self.output_size)
        else:
            self.wiring = FullyConnected(self.hidden_size, output_dim=self.output_size)
        self.ltc = LTCCell(self.wiring,self.input_size)
        self.if_lstm = if_lstm
        if self.if_lstm:
            self.lstm_cell = nn.LSTMCell(self.input_size,self.hidden_size,bias=False)
            self.c_encoder = nn.Linear(self.input_size,self.hidden_size,bias=False)
        self.start_encoder = torch.nn.Linear(self.nav_space,self.hidden_size,bias=True)

        self.optimizer = SophiaG(self.parameters(), lr=lr, weight_decay=weight_decay)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x, raw=False, inference=False):
        x = x.squeeze(-1)
        b, t, _ = x[:,1:,:].shape
        hidden = self.start_encoder(x[:,0,:])
        cell = self.c_encoder(x[:,0,:])
        x_out = []
        if not inference:
            self.hts = torch.zeros(t+1, b, self.hidden_size)
            self.hts[0] = hidden
            for i in range(t):
                if self.if_lstm:
                    hx, cx = self.lstm_cell(x[:, i+1,:], (hidden,cell))
                y, hx = self.ltc(x[:, i+1,:], hidden)  # (batch_size, hidden_size)
                hidden = hx
                cell = cx
                self.hts[i+1] = hx
                x_out.append(y) # (batch_size, output_size)    
        else:
            with torch.no_grad():
                self.hts = np.zeros((t+1, b, self.hidden_size))
                self.hts[0] = hidden.cpu().detach().numpy()
                for i in range(t):
                    if self.if_lstm:
                        hx, cx = self.lstm_cell(x[:, i+1,:], (hidden,cell))
                    y, hx = self.ltc(x[:, i+1,:], hidden)  # (batch_size, hidden_size)
                    hidden = hx
                    cell = cx
                    self.hts[i+1] = hx.cpu().detach().numpy()
                    x_out.append(y.cpu().detach().numpy()) # (batch_size, output_size)

        self.time_steps = t
        self.batch_size = b
        # output shape (batch_size, seq_len, input_size)
        if not inference:
            if not raw:
                return torch.stack(x_out, dim=0).permute(1, 0, 2)
        else:
            if not raw:
                return np.array(x_out).transpose(1, 0, 2)
        return self.hts
    
class LTC_solver_0start(LTC_solver):
    def __init__(self,input_size,hidden_size,output_size=2,lr=0.0002,act_decay=1.0,weight_decay=0.01,irnn=True,outputnn=True,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=2,if_lstm=False,if_wiring=True):
        super().__init__(input_size,hidden_size,output_size=output_size,lr=lr,act_decay=act_decay,weight_decay=weight_decay,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space,if_lstm=if_lstm,if_wiring=if_wiring)

    def forward(self, x, raw=False, inference=False):
        x = x.squeeze(-1)
        b, t, _ = x[:,:,:].shape
        hidden = torch.zeros(b, self.hidden_size)  # default to zeros
        cell = torch.zeros(b, self.hidden_size)  # default to zeros
        x_out = []
        if not inference:
            self.hts = torch.zeros(t+1, b, self.hidden_size)
            self.hts[0] = hidden
            for i in range(t):
                if self.if_lstm:
                    hx, cx = self.lstm_cell(x[:, i,:], (hidden,cell))
                y, hx = self.ltc(x[:, i,:], hidden)  # (batch_size, hidden_size)
                hidden = hx
                cell = cx
                self.hts[i+1] = hx
                x_out.append(y) # (batch_size, output_size)    
        else:
            with torch.no_grad():
                self.hts = np.zeros((t+1, b, self.hidden_size))
                self.hts[0] = hidden.cpu().detach().numpy()
                for i in range(t):
                    if self.if_lstm:
                        hx, cx = self.lstm_cell(x[:, i,:], (hidden,cell))
                    y, hx = self.ltc(x[:, i,:], hidden)  # (batch_size, hidden_size)
                    hidden = hx
                    cell = cx
                    self.hts[i+1] = hx.cpu().detach().numpy()
                    x_out.append(y.cpu().detach().numpy()) # (batch_size, output_size)

        self.time_steps = t
        self.batch_size = b
        # output shape (batch_size, seq_len, input_size)
        if not inference:
            if not raw:
                return torch.stack(x_out, dim=0).permute(1, 0, 2)
        else:
            if not raw:
                return np.array(x_out).transpose(1, 0, 2)
        return self.hts

    
class LTC_solver_Low(LTC_solver):
    def __init__(self,input_size,hidden_size,output_size=4,lr=0.0002,act_decay=1.0,weight_decay=0.01,irnn=True,outputnn=True,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=2,if_lstm=False,if_wiring=True):
        super().__init__(input_size,hidden_size,output_size=output_size,lr=lr,act_decay=act_decay,weight_decay=weight_decay,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space,if_lstm=if_lstm,if_wiring=if_wiring)

    def loss_fn(self, x, y_hat):
        hts = self(x,raw=True)
        # Activity loss
        activity_L2 = self.act_decay/(self.time_steps*self.hidden_size*self.batch_size)*(hts**2).sum()
        y = self(x,raw=False)
        # y_hat = y_hat.transpose(0,1)
        loss_sin_x = self.loss_func(y[:,:,0],y_hat[:,:,0])
        loss_cos_x = self.loss_func(y[:,:,1],y_hat[:,:,1])
        loss_sin_y = self.loss_func(y[:,:,2],y_hat[:,:,2])
        loss_cos_y = self.loss_func(y[:,:,3],y_hat[:,:,3])
        loss = loss_sin_x + loss_cos_x + loss_sin_y + loss_cos_y + activity_L2
        self.losses.append(loss.item())
        return loss
    
    def train_gradual_manual(self,input):
        # Input shape: [Epochs,data/labels,batchsize,tsteps,x/y]
        t = tqdm(range(len(input)), desc="Loss", leave=True)
        for i in t:
            data = input[i][0]
            labels = input[i][1]
            labels = sincos_from_2D(labels)
            loss = self.train_step(data.to(device),labels.to(device))
            t.set_description(f"Loss: {loss:.5f}", refresh=True)

class CfC_solver(LTC_solver):
    def __init__(self,input_size,hidden_size,output_size=2,lr=0.0002,act_decay=1.0,weight_decay=0.01,irnn=True,outputnn=True,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=2,if_lstm=False,if_wiring=True):
        super().__init__(input_size,hidden_size,output_size=output_size,lr=lr,act_decay=act_decay,weight_decay=weight_decay,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space,if_lstm=if_lstm,if_wiring=if_wiring)
        self.cfc = WiredCfCCell(self.input_size,self.wiring)
        self.if_lstm = if_lstm
        if self.if_lstm:
            self.lstm_cell = nn.LSTMCell(self.input_size,self.hidden_size,bias=False)
            self.c_encoder = nn.Linear(self.input_size,self.hidden_size,bias=False)
        self.start_encoder = torch.nn.Linear(self.nav_space,self.hidden_size,bias=True)

        self.optimizer = SophiaG(self.parameters(), lr=lr, weight_decay=weight_decay)
        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    def forward(self, x, raw=False, inference=False):
        x = x.squeeze(-1)
        b, t, _ = x[:,1:,:].shape
        hidden = self.start_encoder(x[:,0,:])
        cell = self.c_encoder(x[:,0,:])
        x_out = []
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
                x_out.append(y) # (batch_size, output_size)    
        else:
            with torch.no_grad():
                self.hts = np.zeros((t+1, b, self.hidden_size))
                self.hts[0] = hidden.cpu().detach().numpy()
                for i in range(t):
                    if self.if_lstm:
                        hx, cx = self.lstm_cell(x[:, i+1,:], (hidden,cell))
                    y, hx = self.cfc(x[:, i+1,:], hidden,timespans=ts)
                    hidden = hx
                    cell = cx
                    self.hts[i+1] = hx.cpu().detach().numpy()
                    x_out.append(y.cpu().detach().numpy()) # (batch_size, output_size)
            
        self.time_steps = t
        self.batch_size = b
        # output shape (batch_size, seq_len, input_size)
        if not inference:
            if not raw:
                return torch.stack(x_out, dim=0).permute(1, 0, 2)
        else:
            if not raw:
                return np.array(x_out).transpose(1, 0, 2)
        return self.hts
    
    def train_gradual_manual(self,input):
        # Input shape: [Epochs,data/labels,batchsize,tsteps,x/y]
        t = tqdm(range(len(input)), desc="Loss", leave=True)
        for i in t:
            data = input[i][0]
            labels = input[i][1]
            loss = self.train_step(data.to(device),labels.to(device))
            t.set_description(f"Loss: {loss:.5f}", refresh=True)
    

class CfC_solver_Low(CfC_solver):
    def __init__(self,input_size,hidden_size,output_size=4,lr=0.0002,act_decay=1.0,weight_decay=0.01,irnn=True,outputnn=True,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=2,if_lstm=False,if_wiring=True):
        super().__init__(input_size,hidden_size,output_size=output_size,lr=lr,act_decay=act_decay,weight_decay=weight_decay,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space,if_lstm=if_lstm,if_wiring=if_wiring)

    def loss_fn(self, x, y_hat):
        hts = self(x,raw=True)
        # Activity loss
        activity_L2 = self.act_decay/(self.time_steps*self.hidden_size*self.batch_size)*(hts**2).sum()
        y = self(x,raw=False)
        # y_hat = y_hat.transpose(0,1)
        loss_sin_x = self.loss_func(y[:,:,0],y_hat[:,:,0])
        loss_cos_x = self.loss_func(y[:,:,1],y_hat[:,:,1])
        loss_sin_y = self.loss_func(y[:,:,2],y_hat[:,:,2])
        loss_cos_y = self.loss_func(y[:,:,3],y_hat[:,:,3])
        loss = loss_sin_x + loss_cos_x + loss_sin_y + loss_cos_y + activity_L2
        self.losses.append(loss.item())
        return loss
    
    def train_gradual_manual(self,input):
        # Input shape: [Epochs,data/labels,batchsize,tsteps,x/y]
        t = tqdm(range(len(input)), desc="Loss", leave=True)
        for i in t:
            data = input[i][0]
            labels = input[i][1]
            labels = sincos_from_2D(labels.squeeze(-1))
            loss = self.train_step(data.to(device),labels.to(device))
            t.set_description(f"Loss: {loss:.5f}", refresh=True)
