import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable, functional
import torch.optim as optim


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
    def __init__(self, input_size,time_steps,output_size,hidden_size,lr=0.001,irnn=False,outputnn=False,bias=False,Wx_normalize=False,activation=False):
        super().__init__()
        self.input_size = input_size
        self.time_steps = time_steps
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.hidden = torch.nn.Linear(hidden_size,hidden_size,bias=bias)
        self.input = torch.nn.Linear(input_size,hidden_size,bias=bias)

        self.outputnn = outputnn
        self.output = torch.nn.Linear(hidden_size,output_size,bias=bias)

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
        loss = self.loss_fn(x, y_hat)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, loader, epochs=100):
        for epoch in tqdm(range(epochs)):
            for i,(inputs,labels) in enumerate(loader):
                loss = self.train_step(inputs,labels)
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
        
class RNN_circular(torch_RNN_full_manual):
    def __init__(self, input_size,time_steps,output_size,hidden_size, act_decay=0.001, w_decay=0.001, lr=0.001,irnn=False,outputnn=False,bias=False,Wx_normalize=False,activation=False, rotation_init=False):
        super().__init__(input_size,time_steps,output_size,hidden_size,lr,irnn,outputnn,bias,Wx_normalize,activation)
        self.w_decay = w_decay
        self.act_decay = act_decay
        # self.loss_func = torch.nn.SmoothL1Loss()

        if rotation_init:
            self.hidden.weight = torch.nn.Parameter(torch.tensor([[0,1],[-1,0]],dtype=torch.float32),requires_grad=True)
        
        self.losses_norm = []

    def forward(self, x):
        # h = self.input(x[:,0,:])
        h = torch.zeros(1, x.size(0), self.hidden_size)
        h[:,:,0] = 1
        # h = torch.zeros(1, x.size(0), self.hidden_size)
        self.hts = torch.zeros(self.time_steps, x.size(0), self.hidden_size)
        self.hts[0] = h
        # Main RNN loop
        for t in range(0,self.time_steps):
            h = self.act(self.hidden(h) + self.input(x[:,t,:]))
            self.hts[t] = h
        # If outputnn is true, use a linear layer to output the hs
        if self.outputnn:
            return self.output(self.hts)
        # Else, output all the angles of the hs
        else:
            # Use remainder to make sure angles are between 0 and 2pi
            return torch.remainder(torch.atan2(self.hts[:,:,1],self.hts[:,:,0]),2*np.pi).T
            # return torch.atan2(self.hts[:,:,1],self.hts[:,:,0]).T

    def loss_fn(self, x, y_hat):
        y = self(x)
        # Angles of the hs
        # angles = torch.atan2(y[:,:,1],y[:,:,0])
        # L2 regularization on norm of hs
        # activity_L2 = (self.act_decay*torch.norm(self.hts,dim=-1)**2).sum()
        # norm of hs at each time step regularized around 1
        activity_L2 = self.act_decay*((torch.norm(self.hts,dim=-1)-1)**2).sum()
        angle_loss = self.loss_func(y,y_hat)
        # output_L2 = (self.w_decay*self.output.weight**2).sum()
        # input_L2 = (self.w_decay*self.input.weight**2).sum()
        hidden_L2 = (self.w_decay*self.hidden.weight**2).sum()
        self.losses.append(angle_loss.item())
        self.losses_norm.append(activity_L2.item())
        loss = angle_loss + activity_L2 + hidden_L2
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