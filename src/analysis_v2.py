import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import umap.umap_ as umap
import plotly.express as px
from sklearn.decomposition import PCA

import sys
sys.path.append('../src')
from datagen import *
from datagen2D import *
from datagen2D_v2 import *


class Analysis:
    def __init__(self,model):
        self.model = model

    def plot_losses(self,average=None):
        task_losses = np.array(self.model.task_losses)
        total_losses = np.array(self.model.total_losses)
        if self.model.act_decay == 0.0:
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
        plt.ylim(0,min(loss_data_avgd.max(),4))
        plt.title("MSE Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        return total_losses, task_losses
    
    def binned_mean_activity(self,t_test=40,test_batch_size=5000, circular=True, bins=2000, in_activity=None,start=0):
        if in_activity is None:
            dg = datagen2D_OU(circular=circular)
            data, labels = dg(test_batch_size,t_test)
            start = data[:,0,:]
        else:
            data, labels = in_activity
            test_batch_size = data.shape[0]
            t_test = data.shape[1]

        if type(start) is torch.Tensor:
            labels = torch.concatenate((start.squeeze().unsqueeze(1),labels),dim=1)
        else:
            # Append start to labels
            labels_old = labels.clone()
            labels = torch.zeros(test_batch_size,t_test+1,2)
            labels[:,0,:] = torch.ones_like(labels[:,0,:])*start
            labels[:,1:,:] = labels_old
        
        
        # Get positions from labels
        xs = labels[0:test_batch_size,:,0]
        if type(xs) is torch.Tensor:
            xs = xs.cpu().detach().numpy().T
        ys = labels[0:test_batch_size,:,1]
        if type(ys) is torch.Tensor:
            ys = ys.cpu().detach().numpy().T

        # Get the hidden states inferenced from the test data
        hts = self.model(data,raw=True,inference=True)
        hts = hts.cpu().detach().numpy() # Shape [t_steps, batch_size, hidden_size] = [21, 64, 128]
        n_cells = hts.shape[2]
        
        import scipy.stats as stats

        activity = np.zeros((n_cells,bins,bins))

        for k in tqdm(range(n_cells)):
            # Make all activity positive
            hts_k = abs(hts[:,:,k])
            # Bins equally spaced from 0 to 1 in time_steps amount of bins
            # bin_means, bin_edges, binnumber = stats.binned_statistic(xs.flatten(),hts_k.flatten(),statistic='mean',bins=bins)
            bin_means, bin_edges_x, bin_edges_y, binnumber = stats.binned_statistic_2d(xs.flatten(),ys.flatten(),hts_k.flatten(),statistic='mean',bins=bins)
            activity[k,:] = bin_means
            np.nan_to_num(activity,copy=False)
        self.activity = activity
        self.bin_edges_x = bin_edges_x
        self.bin_edges_y = bin_edges_y
        return self.activity, self.bin_edges_x, self.bin_edges_y
    
    def plot_2D_activity(self,k_test,scale_to_one=False,more_plots=False,plot_head_frac=1/10):
        activity = self.activity
        xbin_edges = self.bin_edges_x
        ybin_edges = self.bin_edges_y

        if scale_to_one:
            scaler = 1/(2*np.pi)
        else:
            scaler = 1

        xbin_edges = xbin_edges[:-1]*scaler
        ybin_edges = ybin_edges[:-1]*scaler

        n_cells = activity.shape[0]
        cell_scaler = n_cells/128

        if k_test >= n_cells:
            print("k_test is larger than the number of cells. Setting k_test to -1.")
            k_test = n_cells-1

        # Plot heat map of activity of cell k_test
        plt.figure(figsize=(5,5))
        plt.imshow(activity[k_test],extent=[xbin_edges[0],xbin_edges[-1],ybin_edges[0],ybin_edges[-1]],vmin=0,vmax=np.max(activity[k_test]),interpolation="bicubic",cmap="jet")
        plt.colorbar()
        plt.title("Heat map of the activity of cell "+str(k_test))
        plt.xlabel(r"Position $x$")
        plt.ylabel(r"Position $y$")
        plt.show()

        if more_plots:
            fig, ax = plt.subplots(int(32*plot_head_frac*cell_scaler),4)
            fig.set_size_inches(15, 80*plot_head_frac*cell_scaler)
            fig.subplots_adjust(hspace=1,wspace=0.2)
            for k in tqdm(range(int(n_cells*plot_head_frac))):
                ax[k//4,k%4].imshow(activity[k],extent=[xbin_edges[0],xbin_edges[-1],ybin_edges[0],ybin_edges[-1]],vmin=0,vmax=np.max(activity[k]),interpolation="bicubic",cmap="jet")
                ax[k//4,k%4].set_title("Cell "+str(k))
                ax[k//4,k%4].set_xlabel(r"Position $x$")
                ax[k//4,k%4].set_ylabel(r"Position $y$")
            plt.show()




def plot_norm(hts,avg_only=True):
    if avg_only:
        # Plot mean
        plt.plot(np.mean(hts,axis=1), linewidth=3, label="Mean")
    else:
        # Plot all in last batch
        plt.plot(hts, linewidth=0.5, alpha=0.5)
        # Plot mean
        plt.plot(np.mean(hts,axis=1), linewidth=3, label="Mean")
    plt.title("Norm of cells h for data in last trainingbatch")
    plt.xlabel(r"Time step $t$")
    plt.ylabel(r"$||h_t||$")
    plt.legend()
    plt.show()
    print("Mean norm: ", np.mean(hts))
    return np.mean(hts,axis=1)