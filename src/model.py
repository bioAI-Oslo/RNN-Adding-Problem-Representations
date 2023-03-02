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
    def __init__(self, input_size,time_steps,output_size,hidden_size,lr=0.001,irnn=False,outputnn=False,bias=False):
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


        self.outputnn = outputnn
        self.output = torch.nn.Linear(hidden_size,output_size)

        self.loss_func = torch.nn.MSELoss()

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.losses = []
        self.accs = []

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size)
        _, hn = self.rnn(x, h0)
        if self.outputnn:
            return self.output(hn).squeeze()
        return hn.squeeze()

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
                acc = (abs(self(inputs)-labels) <= 0.0025*self.time_steps).float().sum().item()/len(inputs)**2
                self.accs.append(acc)
        return self.losses
    

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