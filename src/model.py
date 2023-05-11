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


class torch_RNN1(nn.Module):
    def __init__(self, input_size,time_steps,output_size,hidden_size,lr=0.001,irnn=False,outputnn=False,bias=False,Wx_normalize=False):
        super().__init__()
        self.input_size = input_size
        self.time_steps = time_steps
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.rnn = nn.RNN(input_size, hidden_size, num_layers=1, batch_first=True,nonlinearity = 'relu',bias=bias)
        # Initialize IRNN
        if irnn:
            torch.nn.init.eye_(self.rnn.weight_hh_l0)
            if bias:
                torch.nn.init.constant_(self.rnn.bias_hh_l0, 0)
                torch.nn.init.constant_(self.rnn.bias_ih_l0, 0)

        if Wx_normalize:
        # Set norm of weights to 1
            self.rnn.weight_ih_l0 = torch.nn.Parameter(self.rnn.weight_ih_l0/torch.norm(self.rnn.weight_ih_l0,dim=0).unsqueeze(1))
            # Freeze weights
            self.rnn.weight_ih_l0.requires_grad = False
            # print('Wx_norm: ',torch.norm(self.rnn.weight_ih_l0,dim=0).unsqueeze(1))

        self.Wh_init = self.rnn.weight_hh_l0.detach().clone()
        self.Wx_init = self.rnn.weight_ih_l0.detach().clone()

        self.outputnn = outputnn
        self.output = torch.nn.Linear(hidden_size,output_size)

        self.loss_func = torch.nn.MSELoss()

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.losses = []
        self.accs = []

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        hts, hn = self.rnn(x, h0)
        self.hts_last = hts
        # If outputnn is true, use a linear layer to output the final hidden state
        if self.outputnn:
            return self.output(hn).squeeze()
        # Else, output the "raw" final hidden state
        elif self.hidden_size == 1:
            return hn.squeeze()
        else:
            return torch.norm(hn.squeeze(),dim=-1)

    def loss_fn(self, x, y_hat):
        y = self(x)
        loss = self.loss_func(y,y_hat)
        return loss

    def train_step(self, x, y_hat):
        self.optimizer.zero_grad()
        loss = self.loss_fn(x, y_hat)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, loader, epochs=100):
        for epoch in tqdm(range(epochs)):
            for i,(inputs,labels) in enumerate(loader):
                loss = self.train_step(inputs,labels)
                self.losses.append(loss)
                # acc = (abs(self(inputs)-labels) <= 0.0025*self.time_steps).float().sum().item()/len(inputs)**2
                # self.accs.append(acc)
        return self.losses
    
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
        plt.title("MSE Losses")
        plt.show()

    def plot_accs(self):
        accs = np.array(self.accs)
        acc_data_avgd = accs.reshape(10,-1).mean(axis=1)
        plt.plot(accs)
        plt.title("Accuracy on training data")
        plt.show()


class torch_RNN_manual(torch_RNN1):
    def __init__(self, input_size,time_steps,output_size,hidden_size,lr=0.001,irnn=False,outputnn=False,bias=False,Wx_normalize=False,activation=False):
        super().__init__(input_size,time_steps,output_size,hidden_size,lr,irnn,outputnn,bias)
        self.hidden = torch.nn.Linear(hidden_size,hidden_size,bias=bias)
        self.input = torch.nn.Linear(input_size,hidden_size,bias=bias)
        self.output = torch.nn.Linear(hidden_size,output_size,bias=bias)

        if Wx_normalize:
        # Set norm of weights to 1
            self.input.weight = torch.nn.Parameter(self.input.weight/torch.norm(self.input.weight,dim=0).unsqueeze(1),requires_grad=True)
            # Freeze weights
            self.input.weight.requires_grad = False
            # print('Wx_norm: ',torch.norm(self.input.weight,dim=0).unsqueeze(1))

        self.activation = activation
        if self.activation:
            self.act = torch.nn.ReLU()
        else:
            self.act = torch.nn.Identity()

        if irnn:
            torch.nn.init.eye_(self.hidden.weight)
            if bias:
                torch.nn.init.constant_(self.hidden.bias, 0)
                torch.nn.init.constant_(self.input.bias, 0)

    def forward(self, x):
        # h = self.input(x[:,0,:])
        h = torch.zeros(1, x.size(0), self.hidden_size)
        self.hts = torch.zeros(self.time_steps, x.size(0), self.hidden_size)
        self.hts[0] = h
        # Main RNN loop
        for t in range(0,self.time_steps):
            h = self.act(self.hidden(h) + self.input(x[:,t,:]))
            self.hts[t] = h
        # If outputnn is true, use a linear layer to output the final hidden state
        if self.outputnn:
            return self.output(h).squeeze()
        # Else, output the "raw" final hidden state
        elif self.hidden_size == 1:
            return h.squeeze()
        else:
            return torch.norm(h.squeeze(),dim=-1)
    

class torch_RNN_timewise(torch_RNN1):
    def __init__(self, input_size,time_steps,output_size,hidden_size,lr=0.001,irnn=False,outputnn=False,bias=False):
        super().__init__(input_size,time_steps,output_size,hidden_size,lr,irnn,outputnn,bias)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        hts, hn = self.rnn(x, h0)
        if self.outputnn:
            return self.output(hts).squeeze()
        return hts.squeeze()

    def loss_fn(self, x, y_hat):
        y = self(x)
        loss = self.loss_func(y,y_hat)
        return loss
    

class torch_RNN_full_manual(nn.Module):
    def __init__(self, input_size,time_steps,output_size,hidden_size,lr=0.001,irnn=False,outputnn=False,bias=False,Wx_normalize=False,activation=False,batch_size=64):
        super().__init__()
        self.input_size = input_size
        self.time_steps = time_steps
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.hidden = torch.nn.Linear(hidden_size,hidden_size,bias=bias)
        self.input = torch.nn.Linear(input_size,hidden_size,bias=bias)

        self.outputnn = outputnn
        self.output = torch.nn.Linear(hidden_size,output_size,bias=bias)

        # Make a trainable bias for the hts to potentially move the center of the circle
        # self.hts_bias = torch.nn.Parameter(torch.zeros(self.time_steps+1, self.hidden_size),requires_grad=True)
        self.hts_bias = torch.nn.Parameter(torch.zeros(self.hidden_size),requires_grad=True)
        # Make a trainable starting hidden state / starting points of the circle
        self.h0 = torch.nn.Parameter(torch.ones(1, self.hidden_size)/np.sqrt(self.hidden_size),requires_grad=True)

        # self.h0 = torch.zeros(1, self.hidden_size)
        # self.h0[0,0] = 1

        self.batch_size = batch_size

        # Initialize IRNN
        self.activation = activation
        if self.activation:
            self.act = torch.nn.ReLU()
        else:
            self.act = torch.nn.Identity()

        if irnn:
            torch.nn.init.eye_(self.hidden.weight)
            if bias:
                torch.nn.init.constant_(self.hidden.bias, 0)
                torch.nn.init.constant_(self.input.bias, 0)

        if Wx_normalize:
        # Set norm of weights to 1
            self.input.weight = torch.nn.Parameter(self.input.weight/torch.norm(self.input.weight,dim=0).unsqueeze(1),requires_grad=True)
            # Freeze weights
            self.input.weight.requires_grad = False
            # print('Wx_norm: ',torch.norm(self.input.weight,dim=0).unsqueeze(1))

        self.Wh_init = self.hidden.weight.detach().clone()
        self.Wx_init = self.input.weight.detach().clone()

        self.loss_func = torch.nn.MSELoss()

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.losses = []
        self.accs = []

    def forward(self, x):
        # h = self.input(x[:,0,:])
        h = torch.zeros(1, x.size(0), self.hidden_size)
        self.hts = torch.zeros(self.time_steps, x.size(0), self.hidden_size)
        self.hts[0] = h
        # Main RNN loop
        for t in range(0,self.time_steps):
            h = self.act(self.hidden(h) + self.input(x[:,t,:]))
            self.hts[t] = h
        # If outputnn is true, use a linear layer to output the final hidden state
        if self.outputnn:
            return self.output(h).squeeze()
        # Else, output the "raw" final hidden state
        elif self.hidden_size == 1:
            return h.squeeze()
        else:
            return torch.norm(h.squeeze(),dim=-1)

    def loss_fn(self, x, y_hat):
        y = self(x)
        loss = self.loss_func(y,y_hat)
        self.losses.append(loss.item())
        return loss

    def train_step(self, x, y_hat):
        self.optimizer.zero_grad(set_to_none=True)
        # Backward hook to clip gradients
        loss = self.loss_fn(x, y_hat)
        loss.backward()
        self.optimizer.step()
        # Print weight gradient norms
        # print("Hidden weight grad norm:",torch.norm(self.hidden.weight.grad))
        return loss.item()

    def train(self, epochs=100, loader=None):
        for epoch in tqdm(range(epochs)):
            data,labels = datagen_circular(self.batch_size,self.time_steps)
            loss = self.train_step(data,labels)
            # for i,(inputs,labels) in enumerate(loader):
            #     loss = self.train_step(inputs,labels)
                # self.losses.append(loss)
                # acc = (abs(self(inputs)-labels) <= 0.0025*self.time_steps).float().sum().item()/len(inputs)**2
                # self.accs.append(acc)
        return self.losses
    
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
        plt.title("MSE Losses")
        plt.show()

    def plot_accs(self):
        accs = np.array(self.accs)
        acc_data_avgd = accs.reshape(10,-1).mean(axis=1)
        plt.plot(accs)
        plt.title("Accuracy on training data")
        plt.show()


class RNN_L2(torch_RNN_full_manual):
    def __init__(self, input_size,time_steps,output_size,hidden_size, act_decay=0.001, w_decay=0.001, lr=0.001,irnn=False,outputnn=False,bias=False,Wx_normalize=False,activation=False):
        super().__init__(input_size,time_steps,output_size,hidden_size,lr,irnn,outputnn,bias,Wx_normalize,activation)
        self.w_decay = w_decay
        self.act_decay = act_decay

    def loss_fn(self, x, y_hat):
        y = self(x)
        # L2 regularization on hts
        activity_L2 = (self.act_decay*self.hts**2).sum()
        # (h_t - h_{t-1})^2
        # activity_L2 = self.act_decay*torch.tensor([((torch.norm(self.hts[i],dim=1)-torch.norm(self.hts[i-1],dim=1))**2).sum() for i in range(1,len(self.hts))],requires_grad=True).sum()
        output_L2 = (self.w_decay*self.output.weight**2).sum()
        input_L2 = (self.w_decay*self.input.weight**2).sum()
        hidden_L2 = (self.w_decay*self.hidden.weight**2).sum()
        loss = self.loss_func(y,y_hat) + activity_L2 + output_L2 + input_L2 + hidden_L2
        return loss
    
    def forward(self, x):
        # h = self.input(x[:,0,:])
        h = torch.zeros(1, x.size(0), self.hidden_size)
        self.hts = torch.zeros(self.time_steps, x.size(0), self.hidden_size)
        self.hts[0] = h
        # Main RNN loop
        for t in range(0,self.time_steps):
            h = self.act(self.hidden(h) + self.input(x[:,t,:]))
            self.hts[t] = h
        # If outputnn is true, use a linear layer to output the final hidden state
        if self.outputnn:
            return torch.sigmoid(self.output(h).squeeze())
        # Else, output the "raw" final hidden state
        elif self.hidden_size == 1:
            return h.squeeze()
        else:
            return torch.norm(h.squeeze(),dim=-1)


class RNN_circular_2D(torch_RNN_full_manual):
    # RNN that trains to output the angle of a circular trajectory in the first two dimensions depending on size of input, uses atan2 instead of arccos in ND case
    def __init__(self, input_size,time_steps,output_size,hidden_size, act_decay=0.001, w_decay=0.001, lr=0.001,irnn=False,outputnn=False,bias=False,Wx_normalize=False,activation=False, rotation_init=False):
        super().__init__(input_size,time_steps,output_size,hidden_size,lr,irnn,outputnn,bias,Wx_normalize,activation)
        self.w_decay = w_decay
        self.act_decay = act_decay
        if rotation_init:
            # self.hidden.weight = torch.nn.Parameter(torch.tensor([[0,1],[-1,0]],dtype=torch.float32),requires_grad=True)
            self.hidden.weight = torch.nn.Parameter(torch.tensor([[np.cos(2*np.pi/20),-np.sin(2*np.pi/20)],[np.sin(2*np.pi/20),np.cos(2*np.pi/20)]],dtype=torch.float32),requires_grad=True)

        self.losses_norm = []

    def forward(self, x):
        # Make h0 trainable
        h =  self.h0.unsqueeze(1).repeat(1,x.size(0),1)
        # time_steps+1 because we want to include the initial hidden state
        self.hts = torch.zeros(self.time_steps+1, x.size(0), self.hidden_size)
        self.hts[0] = h
        # Main RNN loop
        for t in range(0,self.time_steps):
            h = self.act(self.hidden(h) + self.input(x[:,t,:]))
            self.hts[t+1] = h
        # If outputnn is true, use a linear layer to output the hs
        if self.outputnn:
            return self.output(self.hts)
        # Else, output all the angles of the hs
        else:
            self.hts = self.hts - self.hts_bias.unsqueeze(1)
            # Use remainder to make sure angles are between 0 and 2pi. Note this is only for the first TWO dimensions, and for hdims > 2, this will not be a general angle
            return torch.remainder(torch.atan2(self.hts[1:,:,1],self.hts[1:,:,0]),2*np.pi).T

    def loss_fn(self, x, y_hat):
        y = self(x)
        # L2 regularization on norm of hs
        # norm of hs at each time step regularized around 1
        activity_L2 = self.act_decay*((torch.norm(self.hts,dim=-1)-1)**2).sum()
        # angle_loss = self.loss_func(y,y_hat)

        # Loss for equating 0 and 2pi
        angle_loss = ((torch.abs(torch.sin(y)-torch.sin(y_hat)) + torch.abs(torch.cos(y)-torch.cos(y_hat)))**2).mean()

        self.losses.append(angle_loss.item())
        self.losses_norm.append(activity_L2.item())
        loss = angle_loss + activity_L2 #+ hidden_L2
        return loss
    
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


class RNN_circular_ND(torch_RNN_full_manual):
    # RNN that trains to output the angle of a circular trajectory in any dimension size ND depending on size of input
    def __init__(self, input_size,time_steps,output_size,hidden_size, act_decay=0.001, w_decay=0.001, lr=0.001,irnn=False,outputnn=False,bias=False,Wx_normalize=False,activation=False, rotation_init=False):
        super().__init__(input_size,time_steps,output_size,hidden_size,lr,irnn,outputnn,bias,Wx_normalize,activation)
        self.w_decay = w_decay
        self.act_decay = act_decay
        if rotation_init:
            # self.hidden.weight = torch.nn.Parameter(torch.tensor([[0,1],[-1,0]],dtype=torch.float32),requires_grad=True)
            self.hidden.weight = torch.nn.Parameter(torch.tensor([[np.cos(2*np.pi/20),-np.sin(2*np.pi/20)],[np.sin(2*np.pi/20),np.cos(2*np.pi/20)]],dtype=torch.float32),requires_grad=True)

        self.losses_norm = []
        self.losses_circle = []

    def forward(self, x):
        # Make h0 trainable
        h =  self.h0.unsqueeze(1).repeat(1,x.size(0),1)
        # time_steps+1 because we want to include the initial hidden state
        batch_size = x.size(0)
        self.hts = torch.zeros(self.time_steps+1, batch_size, self.hidden_size)
        self.hts[0] = h
        # Main RNN loop
        for t in range(0,self.time_steps):
            h = self.act(self.hidden(h) + self.input(x[:,t,:]))
            self.hts[t+1] = h
        # If outputnn is true, use a linear layer to output the hs
        if self.outputnn:
            return self.output(self.hts)
        # Else, output all the angles of the hs
        else:
            # Bias to potentially move the center of the circle
            self.hts = self.hts #- self.hts_bias.unsqueeze(1)

            # Check if returns NaN before returning
            if torch.acos(torch.clamp(self.hts[1:,:,0]/torch.norm(self.hts[1:,:,:],dim=-1),-1.0,1.0)).T.isnan().any():
                # print("NaN in angle")
                pass

            return self.hts
            # return torch.acos(torch.clamp(self.hts[1:,:,0]/torch.norm(self.hts[1:,:,:],dim=-1),-1.0,1.0)).T
            # return torch.atan2(self.hts[:,:,1],self.hts[:,:,0]).T

    def loss_fn(self, x, y_hat):
        y = self(x)
        # y = y.permute(1,0,2)
        # L2 regularization on norm of hs
        # norm of hts at each time step regularized to be 1
        activity_L2 = self.act_decay*((torch.norm(y,dim=-1)-1)**2).sum()
        

        # Angle loss
        # Concatenate 0 to y_hat to make it the same size as y
        y_hat = torch.cat((torch.zeros(y_hat.size(0),1),y_hat),dim=1)
        # Permute y_hat to make it the same size as y
        y_hat = y_hat.permute(1,0)
        # angle_loss = 0

        # for i in range(1,self.time_steps):
        #     # Iterate through 1 to i followed up to time_steps//2 minus 10% of time_steps as explained in obsidian notes
        #     for j in range(1,min(self.time_steps//2-int(self.time_steps*0.1),i)+1):
        #         # print(y.shape,y_hat.shape)
        #         # Check angles between hts at time i and i-j
        #         # angle_test = torch.abs(torch.acos(torch.clamp(torch.sum(y[i]*y[i-j],dim=-1)/(torch.norm(y[i],dim=-1)*torch.norm(y[i-j],dim=-1)),-1.0,1.0)))
        #         normalizer = 1/(torch.norm(y[i],dim=-1)*torch.norm(y[i-j],dim=-1))
        #         # Check if y has any NaNs
        #         # if y.isnan().any():
        #         #     print("y has NaNs")
        #         # Check if normalizer has any 0s
        #         # if normalizer.isnan().any():
        #         #     print("normalizer has NaNs")
        #         angle_test = torch.abs(torch.acos(torch.clamp(torch.sum(y[i]*y[i-j],dim=-1)*normalizer,-0.9999,0.9999)))
        #         # Check if angle_test has any NaNs
        #         # if angle_test.isnan().any():
        #         #     print("angle_test has NaNs")
        #         angle_theoretical = (y_hat[i]-y_hat[i-j])
        #         angle_loss += torch.mean((angle_test-angle_theoretical)**2)

        
        i = torch.arange(1, self.time_steps).unsqueeze(1)
        j = torch.arange(1, self.time_steps//2-int(self.time_steps*0.1)).unsqueeze(0)
        # CONSIDER CHANGING THIS TO > INSTEAD OF >=
        mask = (i >= j).float()
        j = j * mask
        # Convert i and j to int
        i = i.long()
        j = j.long()
        normalizer = 1 / (torch.norm(y[i], dim=-1) * torch.norm(y[i-j], dim=-1))
        # Cant clamp between -1 and 1 because it will cause NaNs in training
        angle_test = torch.abs(torch.acos(torch.clamp(torch.sum(y[i]*y[i-j], dim=-1) * normalizer, -0.9999, 0.9999)))
        angle_theoretical = (y_hat[i]-y_hat[i-j])
        angle_loss = torch.mean((angle_test-angle_theoretical)**2)

        # Loss to end in the same position as the start
        circle_end_loss = 0.0001*torch.mean((y[-1]-y[0])**2)

        # Mean the angle loss over the batch and time steps
        # angle_loss = angle_loss/x.size(0)

        self.losses.append(angle_loss.item())
        self.losses_norm.append(activity_L2.item())
        self.losses_circle.append(circle_end_loss.item())
        loss = angle_loss + activity_L2 + circle_end_loss
        return loss
    
    def plot_losses(self,average=None):
        losses = np.array(self.losses)
        losses_norm = np.array(self.losses_norm)
        losses_circle = np.array(self.losses_circle)
        if average == None:
            plt.plot(losses[10:],label="Angle Loss")
            plt.plot(losses_norm[10:], label="Activity norm Loss")
            plt.plot(losses_circle[10:], label="Circle End Loss")
            plt.plot(losses[10:]+losses_norm[10:]+losses_circle[10:], label="Total Loss")
        else:
            if len(losses)%average != 0:
                losses = losses[:-(len(losses)%average)]
                losses_norm = losses_norm[:-(len(losses_norm)%average)]
                losses_circle = losses_circle[:-(len(losses_circle)%average)]
                print("Losses array was not a multiple of average. Truncated to",len(losses))
            loss_data_avgd = losses.reshape(average,-1).mean(axis=1)
            loss_data_norm_avgd = losses_norm.reshape(average,-1).mean(axis=1)
            loss_data_circle_avgd = losses_circle.reshape(average,-1).mean(axis=1)
            plt.plot(loss_data_avgd[3:], label="Angle Loss")
            plt.plot(loss_data_norm_avgd[3:], label="Activity norm Loss")
            plt.plot(loss_data_circle_avgd[3:], label="Circle End Loss")
            plt.plot(loss_data_avgd[3:]+loss_data_norm_avgd[3:]+loss_data_circle_avgd[3:], label="Total Loss")
        plt.legend()
        plt.title("MSE Losses")
        plt.show()


class RNN_circular_ND_pm(torch_RNN_full_manual):
    # RNN that trains to sum positive and negative numbers (pm) on a circle in any dimension size ND depending on size of input
    def __init__(self, input_size,time_steps,output_size,hidden_size, act_decay=0.001, w_decay=0.001, lr=0.001,irnn=False,outputnn=False,bias=False,Wx_normalize=False,activation=False, rotation_init=False):
        super().__init__(input_size,time_steps,output_size,hidden_size,lr,irnn,outputnn,bias,Wx_normalize,activation)
        self.w_decay = w_decay
        self.act_decay = act_decay
        if rotation_init:
            # self.hidden.weight = torch.nn.Parameter(torch.tensor([[0,1],[-1,0]],dtype=torch.float32),requires_grad=True)
            self.hidden.weight = torch.nn.Parameter(torch.tensor([[np.cos(2*np.pi/20),-np.sin(2*np.pi/20)],[np.sin(2*np.pi/20),np.cos(2*np.pi/20)]],dtype=torch.float32),requires_grad=True)

        self.losses_norm = []

    def forward(self, x):
        # Make h0 trainable
        h =  self.h0.unsqueeze(1).repeat(1,x.size(0),1)
        self.time_steps = x.size(1)
        # time_steps+1 because we want to include the initial hidden state
        batch_size = x.size(0)
        self.hts = torch.zeros(self.time_steps+1, batch_size, self.hidden_size)
        self.hts[0] = h
        # Main RNN loop
        for t in range(0,self.time_steps):
            h = self.act(self.hidden(h) + self.input(x[:,t,:]))
            self.hts[t+1] = h
        # If outputnn is true, use a linear layer to output the hs
        if self.outputnn:
            return self.output(self.hts)
        # Else, output all hts
        else:
            # Bias to potentially move the center of the circle
            self.hts = self.hts - self.hts_bias.unsqueeze(0).unsqueeze(0)

            # Check if returns NaN before returning
            # if torch.acos(torch.clamp(self.hts[1:,:,0]/torch.norm(self.hts[1:,:,:],dim=-1),-1.0,1.0)).T.isnan().any():
                # print("NaN in angle")
                # pass

            return self.hts

    def loss_fn(self, x, y_hat):
        y = self(x)
        # y = y.permute(1,0,2)
        # L2 regularization on norm of hs
        # norm of hts at each time step regularized to be 1
        activity_L2 = self.act_decay*((torch.norm(y,dim=-1)-1)**2).sum()
        
        # Concatenate 0 (h0) to y_hat to make it the same size as y
        y_hat = torch.cat((torch.zeros(y_hat.size(0),1),y_hat),dim=1)
        # Permute y_hat to make it the same size as y
        y_hat = y_hat.permute(1,0)
        # angle_loss = 0
        
        # Main angle loss loop, checks difference in angles for multiple time steps back in time
        i = torch.arange(1, self.time_steps).unsqueeze(1)
        j = torch.arange(1, self.time_steps//2-int(self.time_steps*0.1)).unsqueeze(0)
        # THIS IS VERY UNCERTAIN, TRY ALSO >= (BUT I THINK IT SHOULD BE >)
        mask = (i > j).float()
        j = j * mask
        # Convert i and j to int
        i = i.long()
        j = j.long()
        normalizer = 1 / (torch.norm(y[i], dim=-1) * torch.norm(y[i-j], dim=-1))
        # Cant clamp between -1 and 1 because it will cause NaNs in training
        angle_test = torch.abs(torch.acos(torch.clamp(torch.sum(y[i]*y[i-j], dim=-1) * normalizer, -0.9999999, 0.9999999)))
        # Must use torch.abs because the angle can be negative, but the angle_test only returns positive angles
        # angle_theoretical = torch.abs(y_hat[i]-y_hat[i-j])
        # Make 0 and 2pi the same angle
        angle_theoretical = torch.min(torch.remainder(2*np.pi-(y_hat[i]-y_hat[i-j]),2*np.pi),torch.remainder(2*np.pi-(y_hat[i-j]-y_hat[i]),2*np.pi))
        angle_loss = torch.mean((angle_test-angle_theoretical)**2)

        # Loss to end in the same position as the start
        # circle_end_loss = 0.0001*torch.mean((y[-1]-y[0])**2)

        self.losses.append(angle_loss.item())
        self.losses_norm.append(activity_L2.item())
        # self.losses_circle.append(circle_end_loss.item())
        loss = angle_loss + activity_L2 # + circle_end_loss
        return loss
    
    def train(self, epochs=100, loader=None):
        for epoch in tqdm(range(epochs)):
            data,labels = datagen_truecircular_pm(self.batch_size,self.time_steps)
            loss = self.train_step(data,labels)
            # for i,(inputs,labels) in enumerate(loader):
            #     loss = self.train_step(inputs,labels)
                # self.losses.append(loss)
                # acc = (abs(self(inputs)-labels) <= 0.0025*self.time_steps).float().sum().item()/len(inputs)**2
                # self.accs.append(acc)
        return self.losses
    
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


class RNN_circular_LowEtAl(nn.Module):
    def __init__(self,input_size,hidden_size,lr=0.001,irnn=True,outputnn=True,bias=True,Wx_normalize=False,activation=True,batch_size=64,nav_space=1):
        super().__init__()
        self.input_size = input_size
        # self.time_steps = time_steps
        self.hidden_size = hidden_size
        self.nav_space = nav_space # Number of navigation dimensions, default 1D, if 2D we want torus

        self.hidden = torch.nn.Linear(hidden_size,hidden_size,bias=False)
        self.input = torch.nn.Linear(input_size,hidden_size,bias=bias)

        self.h0_layer = torch.nn.Linear(self.nav_space,hidden_size,bias=bias)

        self.outputnn = outputnn
        self.output = torch.nn.Linear(hidden_size,2*self.nav_space,bias=bias)

        self.batch_size = batch_size

        self.base_training_tsteps = 20

        # Initialize IRNN
        self.activation = activation
        if self.activation:
            self.act = torch.nn.ReLU()
        else:
            self.act = torch.nn.Identity()

        if irnn:
            torch.nn.init.eye_(self.hidden.weight)
            if bias:
                # torch.nn.init.constant_(self.hidden.bias, 0)
                torch.nn.init.constant_(self.input.bias, 0)

            
        if Wx_normalize:
        # Set norm of weights to 1
            self.input.weight = torch.nn.Parameter(self.input.weight/torch.norm(self.input.weight,dim=0).unsqueeze(1),requires_grad=True)
            # Freeze weights
            self.input.weight.requires_grad = False
            # print('Wx_norm: ',torch.norm(self.input.weight,dim=0).unsqueeze(1))

        self.Wh_init = self.hidden.weight.detach().clone()
        self.Wx_init = self.input.weight.detach().clone()

        self.loss_func = torch.nn.MSELoss()

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0.01)

        self.losses = []
        self.accs = []

    def forward(self, x, raw=False):
        # Make h0 trainable
        batch_size_forward = x.size(0)
        theta0s = torch.rand((batch_size_forward,self.nav_space))*2*np.pi
        h = self.h0_layer(theta0s) # Usikker på denne
        time_steps = x.size(1)
        # time_steps+1 because we want to include the initial hidden state
        self.hts = torch.zeros(time_steps+1, batch_size_forward, self.hidden_size)
        self.hts[0] = h
        # Main RNN loop
        for t in range(0,time_steps):
            h = self.act(self.hidden(h) + self.input(x[:,t,:]))
            self.hts[t+1] = h
        # If outputnn is true, use a linear layer to output the hs
        if not raw:
            return self.output(self.hts)
        # Else, output all hts
        else:
            return self.hts

    def loss_fn(self, x, y_hat):
        y = self(x)[1:,:,:]
        y_hat = y_hat.transpose(0,1)
        loss_sin = self.loss_func(y[:,:,0],y_hat[:,:,0])
        loss_cos = self.loss_func(y[:,:,1],y_hat[:,:,1])
        loss = loss_sin + loss_cos
        self.losses.append(loss.item())
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

    def train(self, epochs=100, loader=None):
        for epoch in tqdm(range(epochs)):
            data,labels, _ = datagen_lowetal(self.batch_size,self.base_training_tsteps)
            loss = self.train_step(data,labels)
        return self.losses
    
    def train_gradual(self, epochs=100, loader=None):
        i = 0
        training_steps = 1
        for epoch in tqdm(range(epochs)):
            if i%50 == 0:
                training_steps += 1
            data,labels,_ = datagen_lowetal(self.batch_size,training_steps)
            loss = self.train_step(data,labels)
            i+=1
        print("Last training time steps:",training_steps)
        return self.losses
    
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
        plt.title("MSE Losses")
        plt.show()

    def plot_accs(self):
        accs = np.array(self.accs)
        acc_data_avgd = accs.reshape(10,-1).mean(axis=1)
        plt.plot(accs)
        plt.title("Accuracy on training data")
        plt.show()

class RNN_circular_LowEtAl_bridged(RNN_circular_LowEtAl):
    # Goal of reducing the model complexity of the parent class
    def __init__(self,input_size,hidden_size,lr=0.001,act_decay=0.01,irnn=True,outputnn=False,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=1):
        super().__init__(input_size,hidden_size,lr=lr,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space)
        self.act_decay = act_decay
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=0)

        self.losses_norm = []

    def forward(self, x, raw=False):
        # Make h0 trainable
        batch_size_forward = x.size(0)
        h = torch.zeros((batch_size_forward,self.hidden_size)) # Gives constant initial hidden state
        h[:,0] = 1
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
        y_hat = torch.cat((torch.zeros(y_hat.size(0),1),y_hat),dim=1)
        # Permute y_hat to make it the same size as y
        y_hat = y_hat.permute(1,0)
        # angle_loss = 0
        
        # Main angle loss loop, checks difference in angles for multiple time steps back in time
        i = torch.arange(1, self.time_steps).unsqueeze(1)
        j = torch.arange(1, self.time_steps//2-int(self.time_steps*0.1)).unsqueeze(0)
        # THIS IS VERY UNCERTAIN, TRY ALSO >= (BUT I THINK IT SHOULD BE >)
        mask = (i > j).float()
        j = j * mask
        # Convert i and j to int
        i = i.long()
        j = j.long()
        normalizer = 1 / (torch.norm(y[i], dim=-1) * torch.norm(y[i-j], dim=-1))
        # Cant clamp between -1 and 1 because it will cause NaNs in training
        angle_test = torch.abs(torch.acos(torch.clamp(torch.sum(y[i]*y[i-j], dim=-1) * normalizer, -0.9999999, 0.9999999)))
        # Must use torch.abs because the angle can be negative, but the angle_test only returns positive angles
        angle_theoretical = torch.abs(y_hat[i]-y_hat[i-j])
        # Make 0 and 2pi the same angle
        # angle_theoretical = torch.min(torch.remainder(2*np.pi-(y_hat[i]-y_hat[i-j]),2*np.pi),torch.remainder(2*np.pi-(y_hat[i-j]-y_hat[i]),2*np.pi))
        angle_loss = torch.mean((angle_test-angle_theoretical)**2)

        # Loss to end in the same position as the start
        # circle_end_loss = 0.0001*torch.mean((y[-1]-y[0])**2)

        self.losses.append(angle_loss.item())
        self.losses_norm.append(activity_L2.item())
        # self.losses_circle.append(circle_end_loss.item())
        loss = angle_loss + activity_L2 # + circle_end_loss
        return loss
    
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
    

class RNN_circular_LowEtAl_bridged_reduced(RNN_circular_LowEtAl_bridged):
    def __init__(self,input_size,hidden_size,lr=0.001,act_decay=0.01,irnn=True,outputnn=False,bias=False,Wx_normalize=False,activation=True,batch_size=64,nav_space=1,time_back=3):
        super().__init__(input_size,hidden_size,lr=lr,act_decay=act_decay,irnn=irnn,outputnn=outputnn,bias=bias,Wx_normalize=Wx_normalize,activation=activation,batch_size=batch_size,nav_space=nav_space)
        self.time_back = time_back

    def loss_fn(self, x, y_hat):
        y = self(x)
        # norm of hts at each time step regularized to be 1
        activity_L2 = self.act_decay*((torch.norm(y,dim=-1)-1)**2).sum()
        
        # Concatenate 0 (h0) to y_hat to make it the same size as y
        y_hat = torch.cat((torch.zeros(y_hat.size(0),1),y_hat),dim=1)
        # Permute y_hat to make it the same size as y
        y_hat = y_hat.permute(1,0)
        # angle_loss = 0
        
        # Main angle loss loop, checks difference in angles for multiple time steps back in time
        i = torch.arange(1, self.time_steps).unsqueeze(1)
        j = torch.arange(1, self.time_back).unsqueeze(0)
        # THIS IS VERY UNCERTAIN, TRY ALSO >= (BUT I THINK IT SHOULD BE >)
        mask = (i > j).float()
        j = j * mask
        # Convert i and j to int
        i = i.long()
        j = j.long()
        normalizer = 1 / (torch.norm(y[i], dim=-1) * torch.norm(y[i-j], dim=-1))
        # Cant clamp between -1 and 1 because it will cause NaNs in training
        angle_test = torch.abs(torch.acos(torch.clamp(torch.sum(y[i]*y[i-j], dim=-1) * normalizer, -0.9999999, 0.9999999)))
        # Must use torch.abs because the angle can be negative, but the angle_test only returns positive angles
        angle_theoretical = torch.abs(y_hat[i]-y_hat[i-j])
        # Make 0 and 2pi the same angle
        # angle_theoretical = torch.min(torch.remainder(2*np.pi-(y_hat[i]-y_hat[i-j]),2*np.pi),torch.remainder(2*np.pi-(y_hat[i-j]-y_hat[i]),2*np.pi))
        angle_loss = torch.mean((angle_test-angle_theoretical)**2)

        self.losses.append(angle_loss.item())
        self.losses_norm.append(activity_L2.item())
        # self.losses_circle.append(circle_end_loss.item())
        loss = angle_loss + activity_L2 # + circle_end_loss
        return loss