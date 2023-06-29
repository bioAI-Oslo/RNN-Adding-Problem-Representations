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


def tuning_curve(model,t_test=40,test_batch_size=5000, bins=2000, spherical_data=True, in_activity=None):
    if spherical_data and in_activity is None:
        data, _, labels = datagen_lowetal(test_batch_size,t_test)
    elif in_activity is not None:
        data, labels = in_activity
    else:
        data,labels = datagen_circular_pm(test_batch_size,t_test,sigma=0.05,bound=0.5)
    
    # Get positions from labels
    xs = labels[0:test_batch_size]
    xs = xs.cpu().detach().numpy().T

    # Get the hidden states inferenced from the test data
    hts = model(data[0:test_batch_size],raw=True)
    hts = hts.cpu().detach().numpy() # Shape [t_steps, batch_size, hidden_size] = [21, 64, 128]
    n_cells = hts.shape[2]
    
    import scipy.stats as stats

    activity = np.zeros((n_cells,bins))

    for k in range(n_cells):
        hts_k = abs(hts[1:,:,k])
        # Bins equally spaced from 0 to 1 in time_steps amount of bins
        bin_means, bin_edges, binnumber = stats.binned_statistic(xs.flatten(),hts_k.flatten(),statistic='mean',bins=bins)
        activity[k,:] = bin_means
        np.nan_to_num(activity,copy=False)
    return activity, bin_edges

def plot_tuning_curve(activity,bin_edges,k,spherical=False,linear=False,plot_head_frac=1/10,scale_to_one=False):
    if scale_to_one:
        scaler = 1/(2*np.pi)
    else:
        scaler = 1

    bin_edges = bin_edges*scaler
    bins = len(bin_edges)

    n_cells = activity.shape[0]

    # Plot single cell
    plt.bar(bin_edges[:-1],activity[k,:],width=6/bins)
    plt.title("Histogram of the activity of cell "+str(k))
    plt.xlabel(r"Position $x$")
    plt.ylabel(r"Activity $h_t$")
    plt.show()

    if spherical:
        fig, ax = plt.subplots(int(32*plot_head_frac),4,subplot_kw={'projection': 'polar'})
        fig.set_size_inches(15, 80*plot_head_frac)
        fig.subplots_adjust(hspace=0.5)
        for k in tqdm(range(int(n_cells*plot_head_frac))):
            ax[k//4,k%4].bar(bin_edges[:-1]*2*np.pi,activity[k,:],width=6/bins,alpha=1)
            ax[k//4,k%4].set_title("Cell "+str(k))
        plt.show()
    
    # Plot all cells linearly
    if linear:
        fig, ax = plt.subplots(int(32*plot_head_frac),4)
        fig.set_size_inches(15, 80*plot_head_frac)
        fig.subplots_adjust(hspace=1,wspace=0.2)
        for k in tqdm(range(int(n_cells*plot_head_frac))):
            ax[k//4,k%4].bar(bin_edges[:-1],activity[k,:],width=6/bins)
            ax[k//4,k%4].set_title("Cell "+str(k))
        plt.show()

def lowD_reduce(activity,if_pca=True,n_components=2,plot=True):
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
        
        reducer = umap.UMAP(n_neighbors=500, n_components=n_components)
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

def plot_accuracy(angs,dy,y_hat,y):
    err = np.abs(angs - dy)
    err_mean = np.mean(err,axis=0)
    err_mean_std = np.std(err,axis=0)
    dy_mean = np.mean(np.abs(dy-0),axis=0)
    dy_mean_std = np.std(np.abs(dy-0),axis=0)
    t_test = err.shape[1]
    # trivial_mean_error = np.mean(np.abs(dy - dy_mean),axis=0)
    plt.plot(dy_mean,label="Trivial 0 guess error")
    plt.plot(err_mean,label="Model Error")
    plt.fill_between(np.arange(t_test),dy_mean-dy_mean_std,dy_mean+dy_mean_std,alpha=0.5, label="Trivial 0 guess error std")
    plt.fill_between(np.arange(t_test),err_mean-err_mean_std,err_mean+err_mean_std,alpha=0.5, label="Model error std")
    plt.legend()
    plt.title("Mean model angle difference from true angle absolute error")
    plt.xlabel("Time Step")
    plt.ylabel("Mean Error (Radians)")
    plt.show()

def plot_path_accuracy(angs,y,example_path=0):
    # Integrate angles to get path
    path = np.zeros((angs.shape[0],angs.shape[1]+1))
    path[:,0] = np.pi
    for i in range(1,angs.shape[1]+1):
        path[:,i] = path[:,i-1] + angs[:,i-1]
    # Calculate path error
    path_err = np.abs(path-y)
    path_err_mean = np.mean(path_err,axis=0)
    path_err_mean_std = np.std(path_err,axis=0)
    t_test = path_err.shape[1]
    # Calculate trivial path error
    y_mean = np.mean(np.abs(y-0),axis=0)
    y_mean_std = np.std(np.abs(y-0),axis=0)
    trivial_path_err = np.abs(path-y_mean)
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

    # Plot example path
    plt.plot(y[example_path,:],label="True Path")
    plt.plot(path[example_path,:],label="Model Path")
    plt.legend()
    plt.title("Example Model Path vs True Path")
    plt.xlabel("Time Step")
    plt.ylabel("Angle (Radians)")
    plt.show()

def tuning_curve_2D(model,t_test=40,test_batch_size=5000, bins=2000, in_activity=None):
    if in_activity is None:
        data, labels = smooth_wandering_2D_squarefix(n_data=test_batch_size,t_steps=t_test,bound=0.5,v_sigma=0.01,d_sigma=0.1,v_bound_reduction=0.15,stability=0.01)
    else:
        data, labels = in_activity
        test_batch_size = data.shape[0]
        t_test = data.shape[1]
    
    
    # Get positions from labels
    xs = labels[0:test_batch_size,:,0]
    if type(xs) is torch.Tensor:
        xs = xs.cpu().detach().numpy().T
    ys = labels[0:test_batch_size,:,1]
    if type(ys) is torch.Tensor:
        ys = ys.cpu().detach().numpy().T

    # Get the hidden states inferenced from the test data
    hts_x = model(data[0:test_batch_size,:,0],raw=True)
    hts_x = hts_x.cpu().detach().numpy() # Shape [t_steps, batch_size, hidden_size] = [21, 64, 128]
    hts_y = model(data[0:test_batch_size,:,1],raw=True)
    hts_y = hts_y.cpu().detach().numpy() # Shape [t_steps, batch_size, hidden_size] = [21, 64, 128]
    n_cells = hts_x.shape[2]
    
    import scipy.stats as stats

    activity = np.zeros((n_cells,bins,bins))

    for k in tqdm(range(n_cells)):
        # Make all activity positive
        hts_xk = abs(hts_x[1:,:,k])
        hts_yk = abs(hts_y[1:,:,k])
        # Bins equally spaced from 0 to 1 in time_steps amount of bins
        # bin_means, bin_edges, binnumber = stats.binned_statistic(xs.flatten(),hts_k.flatten(),statistic='mean',bins=bins)
        bin_means, bin_edges_x, bin_edges_y, binnumber = stats.binned_statistic_2d(xs.flatten(),ys.flatten(),hts_xk.flatten() + hts_yk.flatten(),statistic='mean',bins=bins)
        activity[k,:] = bin_means
        np.nan_to_num(activity,copy=False)
    return activity, bin_edges_x, bin_edges_y

def tuning_curve_23D(model,in_activity,bins=2000):
    data, labels = in_activity
    test_batch_size = data.shape[0]
    t_test = data.shape[1]
    
    # Get positions from labels
    xs = labels[0:test_batch_size,:,0]
    if type(xs) is torch.Tensor:
        xs = xs.cpu().detach().numpy().T
    ys = labels[0:test_batch_size,:,1]
    if type(ys) is torch.Tensor:
        ys = ys.cpu().detach().numpy().T

    # Convert to 0,60,120 degree decomposition
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

    # Get the hidden states inferenced from the test data for each dimension/decomposition
    hts_x = model(data_23D[0:test_batch_size,:,0],raw=True)
    hts_x = hts_x.cpu().detach().numpy() # Shape [t_steps, batch_size, hidden_size] = [21, 64, 128]
    hts_y = model(data_23D[0:test_batch_size,:,1],raw=True)
    hts_y = hts_y.cpu().detach().numpy() # Shape [t_steps, batch_size, hidden_size] = [21, 64, 128]
    hts_z = model(data_23D[0:test_batch_size,:,2],raw=True)
    hts_z = hts_z.cpu().detach().numpy() # Shape [t_steps, batch_size, hidden_size] = [21, 64, 128]
    n_cells = hts_x.shape[2]
    
    import scipy.stats as stats

    activity = np.zeros((n_cells,bins,bins))

    for k in tqdm(range(n_cells)):
        # Make all activity positive
        hts_xk = abs(hts_x[1:,:,k])
        hts_yk = abs(hts_y[1:,:,k])
        hts_zk = abs(hts_z[1:,:,k])
        # Bins equally spaced from 0 to 1 in time_steps amount of bins
        # bin_means, bin_edges, binnumber = stats.binned_statistic(xs.flatten(),hts_k.flatten(),statistic='mean',bins=bins)
        bin_means, bin_edges_x, bin_edges_y, binnumber = stats.binned_statistic_2d(xs.flatten(),ys.flatten(),hts_xk.flatten() + hts_yk.flatten() + hts_zk.flatten(),statistic='mean',bins=bins)
        activity[k,:] = bin_means
        np.nan_to_num(activity,copy=False)
    return activity, bin_edges_x, bin_edges_y

def tuning_curve_2D_fullmodel(model,t_test=40,test_batch_size=5000, bins=2000, in_activity=None,start=0):
    if in_activity is None:
        data, labels = smooth_wandering_2D_squarefix(n_data=test_batch_size,t_steps=t_test,bound=0.5,v_sigma=0.01,d_sigma=0.1,v_bound_reduction=0.15,stability=0.01)
    else:
        data, labels = in_activity
        test_batch_size = data.shape[0]
        t_test = data.shape[1]

    # Append 0,0 to labels
    labels_old = labels
    labels = torch.ones(test_batch_size,t_test+1,2)*start
    labels[:,1:,:] = labels_old
    
    # Get positions from labels
    xs = labels[0:test_batch_size,:,0]
    if type(xs) is torch.Tensor:
        xs = xs.cpu().detach().numpy().T
    ys = labels[0:test_batch_size,:,1]
    if type(ys) is torch.Tensor:
        ys = ys.cpu().detach().numpy().T

    # Get the hidden states inferenced from the test data
    hts = model(data,raw=True)
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
    return activity, bin_edges_x, bin_edges_y

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

def plot_2D_tuning_curve_2(activity,xbin_edges,ybin_edges,k_test,scale_to_one=False,more_plots=False,plot_head_frac=1/10):
    if scale_to_one:
        scaler = 1/(2*np.pi)
    else:
        scaler = 1

    xbin_edges = xbin_edges[:-1]*scaler
    ybin_edges = ybin_edges[:-1]*scaler

    n_cells = activity.shape[0]
    cell_scaler = n_cells/128

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