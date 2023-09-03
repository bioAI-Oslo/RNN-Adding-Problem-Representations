import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
import umap.umap_ as umap
import plotly.express as px
from sklearn.decomposition import PCA

import sys
sys.path.append('../src')
from datagen2D_v2 import *


class Analysis:
    def __init__(self,model,if_low):
        self.model = model
        self.if_low = if_low

    def test_dataset(self,test_batch_size=5000,t_test=40,circular=False):
        dg = datagen2D_OU(circular=circular)
        data, labels = dg(test_batch_size,t_test)
        return data, labels

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
    
    def binned_mean_activity(self,t_test=40,test_batch_size=5000, circular=True, bins=50, in_activity=None,start=0):
        if in_activity is None:
            data,labels = self.test_dataset(test_batch_size,t_test,circular=circular)
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
        return self.activity
    
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

    def plot_norm(self,avg_only=True,skip_first=False):
        if skip_first:
            hts = self.model.hts.norm(dim=2)[:,1:]
        else:
            hts = self.model.hts.norm(dim=2)
        hts = hts.cpu().detach().numpy()
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
    
    def lowD_reduce(self,if_pca=True,n_components=2,plot=True,n_neighbors=500):
        # Flatten
        activity = self.activity.reshape(self.activity.shape[0],-1)
        xcol = np.arange(0,int(np.sqrt(activity.shape[-1])))
        ycol = np.arange(0,int(np.sqrt(activity.shape[-1])))

        xx, yy = np.meshgrid(xcol,ycol)
        cols = xx + yy
        cols_flat = cols.flatten()
        if if_pca:

            reducer = PCA(n_components=n_components)
            reducer.fit(activity.T)
            embedding = reducer.transform(activity.T)
            print(embedding.shape)
            # Explained variance
            print(f"Explained variance for PCA with {n_components} components: {100*np.sum(reducer.explained_variance_ratio_):.3f} %")
            if plot and n_components==2:
                plt.scatter(embedding[:,0],embedding[:,1],s=10)
                plt.gca().set_aspect('equal', 'datalim')
                plt.title('PCA projection of the activity of the grid cells', fontsize=12)
                plt.show()
            else:
                fig = px.scatter_3d(embedding, x=0, y=1, z=2,opacity=0.4,color=cols_flat)
                fig.show()
        else:
            # UMAP
            
            reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components)
            reducer.fit(activity.T)
            embedding = reducer.transform(activity.T)
            print(embedding.shape)
            if plot and n_components==2:
                plt.scatter(embedding[:,0],embedding[:,1],s=10)
                plt.gca().set_aspect('equal', 'datalim')
                plt.title('UMAP projection of the activity of the grid cells', fontsize=12)
                plt.show()
            else:
                fig = px.scatter_3d(embedding, x=0, y=1, z=2,opacity=0.4,color=cols_flat)
                fig.show()

        return embedding, reducer
    
    def eval(self,test_batch_size=1000,t_test=100,plot_example=True):
        data,labels = self.test_dataset(test_batch_size,t_test,circular=False)
        if self.if_low:
            labels_sincos = sincos_from_2D(labels)
            out = self.model(data,inference=True)
            out = out.permute(1,0,2)
            out_decoded = sincos_to_2D(torch.Tensor(out))
            labels = labels.detach().cpu().numpy()
            labels_sincos = labels_sincos.detach().cpu().numpy()
            out_decoded = out_decoded[:,1:,:].detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            # total_mse = np.mean((labels_sincos-out)**2)
            total_mse = np.mean((labels-out_decoded)**2)
        else:
            out = self.model(data, inference=True,raw=False)
            out = out.permute(1,0,2)
            labels = labels.detach().cpu().numpy()
            out = out.detach().cpu().numpy()
            total_mse = np.mean((labels-out)**2)

        print("Total MSE: ",total_mse)

        if plot_example:
            n = 0
            plt.plot(labels[n,:,0],labels[n,:,1],"--o", alpha=0.5, label="Theoretical")
            if self.if_low:
                plt.plot(out_decoded[n,:,0],out_decoded[n,:,1],"--o", alpha=0.5, label="Predicted")
            else:
                plt.plot(out[n,:,0],out[n,:,1],"--o", alpha=0.5, label="Predicted")
            plt.legend()
            plt.xlim(-np.pi,np.pi)
            plt.ylim(-np.pi,np.pi)
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Example Trajectory")
            plt.show()

        # Calculate path error
        if self.if_low:
            path = out_decoded
        else:
            path = out
        y = labels
        path_err = np.sqrt(np.abs(path[:,:,0]-y[:,:,0])**2 + np.abs(path[:,:,0]-y[:,:,0])**2)
        path_err_mean = np.mean(path_err,axis=0)
        path_err_mean_std = np.std(path_err,axis=0)
        t_test = path_err.shape[1]
        # Calculate trivial path error
        y_mean = np.mean(np.abs(y-0),axis=0)
        y_mean_std = np.std(np.abs(y-0),axis=0)
        trivial_path_err = np.sqrt(np.abs(path[:,:,0]-y_mean[:,0]) + np.abs(path[:,:,1]-y_mean[:,1]))
        trivial_path_err_mean = np.mean(trivial_path_err,axis=0)
        trivial_path_err_mean_std = np.std(trivial_path_err,axis=0)
        # Print mean error
        print("Mean error: ",np.mean(path_err))
        print("Mean end error: ",path_err_mean[-1])
        # Plot
        plt.plot(trivial_path_err_mean,label="Trivial 0 guess error")
        plt.plot(path_err_mean,label="Model Error")
        plt.fill_between(np.arange(t_test),trivial_path_err_mean-trivial_path_err_mean_std,trivial_path_err_mean+trivial_path_err_mean_std,alpha=0.5, label="Trivial 0 guess error std")
        plt.fill_between(np.arange(t_test),path_err_mean-path_err_mean_std,path_err_mean+path_err_mean_std,alpha=0.5, label="Model error std")
        plt.legend()
        plt.title("Mean model path difference from true path absolute error")
        plt.xlabel("Time Step")
        plt.ylabel("Mean Error (Radians)")
        plt.show()
        
        return total_mse, path_err_mean, path_err_mean_std


def convert_2D_23D(data,labels):
    # Convert to 0,60,120 degree decomposition from 0,90 degree decomposition
    test_batch_size = data.shape[0]
    t_test = data.shape[1]
    
    basis1 = torch.tensor([np.cos(0), np.sin(0)])
    basis2 = torch.tensor([np.cos(np.pi/3), np.sin(np.pi/3)])
    basis3 = torch.tensor([np.cos(2*np.pi/3), np.sin(2*np.pi/3)])
    basises = torch.stack([basis1, basis2, basis3], dim=0)
    # Scale the decompositioned data so that they can be reconstructed to the original data
    basis_scale = 2/3
    data_23D = torch.zeros(test_batch_size,t_test,3)
    labels_23D = torch.zeros(test_batch_size,t_test,3)
    for i,basis in enumerate(basises):
        data_23D[:,:,i] = torch.sum(data.squeeze()*basis,dim=2)*basis_scale
        labels_23D[:,:,i] = torch.sum(labels*basis,dim=2)*basis_scale

    data_23D = data_23D.unsqueeze(-1)
    return data_23D, labels_23D, basises


### For Arccos loss analysis

def test_angle_inference(model,reducer,t_test=40,test_batch_size=1000,in_activity=None,start=np.pi):
    # Same datagen as in training
    if in_activity is None:
        data,labels = datagen_circular_pm(test_batch_size,t_test,sigma=0.05,bound=0.5)
    else:
        data, labels = in_activity
        t_test = data.shape[1]
        test_batch_size = data.shape[0]

    # Inference model to get the hidden states
    y_hat = model(data[0:test_batch_size],raw=True)
    y_hat = y_hat.permute(1,0,2)
    y_hat = y_hat.cpu().detach().numpy()

    # PCA projection of the hidden states to 2D space for calculation of angles
    y_hat_pca = np.zeros((test_batch_size,t_test+1,2))
    for i in range(test_batch_size):
        y_hat_pca[i,:,:] = reducer.transform(y_hat[i,:,:])
    y_hat = y_hat_pca

    # Get theoretical angles from labels
    y = labels[0:test_batch_size]
    y = y.cpu().detach().numpy()
    # Concatenate pi to the start of y to represent starting angle
    y = np.concatenate((np.ones((test_batch_size,1))*start,y),axis=-1)

    # Error between predicted and theoretical angles
    err = np.zeros((test_batch_size,t_test))
    # Angles predicted by the model
    angs = np.zeros((test_batch_size,t_test))
    # Theoretical angle differences
    dy = np.zeros((test_batch_size,t_test))

    # Calculate angle differences from cosine distance
    for j in range(test_batch_size):
        for i in range(1,t_test+1):
            y_hat_i_normalized = y_hat[j,i]/np.linalg.norm(y_hat[j,i])
            y_hat_i_minus_1_normalized = y_hat[j,i-1]/np.linalg.norm(y_hat[j,i-1])
            # Angle between y_hat[j,i] and y_hat[j,i-1]
            ang = np.arccos(y_hat_i_normalized @ y_hat_i_minus_1_normalized)
            # Use cross product to determine direction/sign of angle change
            angle_direction = np.sign(np.cross(y_hat_i_normalized,y_hat_i_minus_1_normalized))
            angs[j,i-1] = ang*angle_direction
            
            # Angle differance from labels
            dy[j,i-1] = y[j,i]-y[j,i-1]

    # For some reason we have to scale angles by 2*pi
    angs = angs*(2*np.pi)
    # Choose the sign of the angles that gives the smallest error
    ang_plus = angs
    ang_minus = -angs
    err_plus = np.abs(dy-ang_plus)
    err_minus = np.abs(dy-ang_minus)
    if np.mean(err_plus) < np.mean(err_minus):
        err = err_plus
        angs = ang_plus
    else:
        err = err_minus
        angs = ang_minus

    print("Mean error: ",np.mean(np.abs(angs-dy)))
    return angs, dy, err, y_hat, y

def test_angle_inference_x1D(model,reducer,t_test=40,test_batch_size=1000,in_activity=None,start=np.pi,dim="x"):
    # Same datagen as in training
    if in_activity is None:
        data,labels = datagen_circular_pm(test_batch_size,t_test,sigma=0.05,bound=0.5)
    else:
        data, labels = in_activity
        t_test = data.shape[1]
        test_batch_size = data.shape[0]

    # Inference model to get the hidden states
    y_hat = model(data[0:test_batch_size],raw=True,inference=True)
    start = labels[0:test_batch_size][:,0,0].cpu().detach().numpy()
    if dim == "x":
        y_hat = y_hat[:,:,y_hat.shape[-1]//2:]
    elif dim == "y":
        y_hat = y_hat[:,:,:y_hat.shape[-1]//2]
    elif dim == "0":
        y_hat = y_hat[:,:,y_hat.shape[-1]//3:]
    elif dim == "60":
        y_hat = y_hat[:,:,y_hat.shape[-1]//3:2*y_hat.shape[-1]//3]
    elif dim == "120":
        y_hat = y_hat[:,:,:2*y_hat.shape[-1]//3]
    else:
        raise Exception("dim must be x, y, 0, 60 or 120")
    y_hat = y_hat.transpose(1,0,2)
    # y_hat = y_hat.cpu().detach().numpy()

    # PCA projection of the hidden states to 2D space for calculation of angles
    y_hat_pca = np.zeros((test_batch_size,t_test,2))
    for i in range(test_batch_size):
        y_hat_pca[i,:,:] = reducer.transform(y_hat[i,:,:])
    y_hat = y_hat_pca

    # Get theoretical angles from labels
    if dim == "x":
        y = labels[0:test_batch_size][:,:,0]
    elif dim == "y":
        y = labels[0:test_batch_size][:,:,1]
    elif dim == "0":
        y = labels[0:test_batch_size][:,:,0]
    elif dim == "60":
        y = labels[0:test_batch_size][:,:,1]
    elif dim == "120":
        y = labels[0:test_batch_size][:,:,2]
    y = y.cpu().detach().numpy()
    # Concatenate pi to the start of y to represent starting angle
    y = np.concatenate((np.reshape(start, (start.shape[0],1)),y),axis=-1)

    # Error between predicted and theoretical angles
    err = np.zeros((test_batch_size,t_test))
    # Angles predicted by the model
    angs = np.zeros((test_batch_size,t_test))
    # Theoretical angle differences
    dy = np.zeros((test_batch_size,t_test))

    # Calculate angle differences from cosine distance
    for j in range(test_batch_size):
        for i in range(1,t_test):
            y_hat_i_normalized = y_hat[j,i]/np.linalg.norm(y_hat[j,i])
            y_hat_i_minus_1_normalized = y_hat[j,i-1]/np.linalg.norm(y_hat[j,i-1])
            # Angle between y_hat[j,i] and y_hat[j,i-1]
            ang = np.arccos(y_hat_i_normalized @ y_hat_i_minus_1_normalized)
            # Use cross product to determine direction/sign of angle change
            angle_direction = np.sign(np.cross(y_hat_i_normalized,y_hat_i_minus_1_normalized))
            angs[j,i-1] = ang*angle_direction
            
            # Angle differance from labels
            dy[j,i-1] = y[j,i]-y[j,i-1]

    # For some reason we have to scale angles by 2*pi
    angs = angs*(2*np.pi)
    # Choose the sign of the angles that gives the smallest error
    ang_plus = angs
    ang_minus = -angs
    err_plus = np.abs(dy-ang_plus)
    err_minus = np.abs(dy-ang_minus)
    if np.mean(err_plus) < np.mean(err_minus):
        err = err_plus
        angs = ang_plus
    else:
        err = err_minus
        angs = ang_minus

    print("Mean error: ",np.mean(np.abs(angs-dy)))
    return angs, dy, err, y_hat, y