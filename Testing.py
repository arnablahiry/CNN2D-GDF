"""
CNN2D-GDF Testing Pipeline
==========================

This notebook provides a comprehensive testing pipeline for a 2D Convolutional Neural Network 
designed to analyze Gaussian Density Fields (GDF). The pipeline includes:

1. Data loading and preprocessing
2. Model loading and evaluation
3. Performance visualization
4. Statistical analysis of predictions

The model predicts cosmological parameter A from 2D maps generated from Gaussian random fields.
"""

# =============================================================================
# IMPORTS AND DEPENDENCIES
# =============================================================================

# Dataset and Network imports
from Dataset import *
from Network import *

# Standard libraries
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import optuna
import os
import time

# PyTorch and GPU setup
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.backends.cudnn as cudnn

# Statistical analysis
from sklearn.metrics import mean_squared_error
from scipy.stats import norm

# =============================================================================
# CONFIGURATION AND PARAMETERS
# =============================================================================

def setup_analysis_parameters():
    """
    Configure analysis parameters and modes for the testing pipeline.
    
    Returns:
        dict: Dictionary containing all configuration parameters
    """
    config = {
        'n_samples': 100000,  # Total number of samples in the dataset
        'analysis_mode': 'original',  # Options: 'original', 'density', 'top-hat-k'
        'case': 'original',  # Type of cuts on GDF maps
        'A_true': None,  # Fixed cosmological parameter A (None for uniform sampling [0.8,1.2])
        'batch_size': 128,
        'test_hist_samples': 10000,  # Samples for histogram analysis
    }
    
    # Configure analysis-specific parameters
    if config['analysis_mode'] == 'original':
        config.update({
            'dens_case': 'original',
            'dens_cut_str': None,
            'kmax_cut_str': None
        })
    elif config['analysis_mode'] == 'density':
        config.update({
            'dens_case': 'max',  # or 'min'
            'dens_cut_str': '0.5',
            'kmax_cut_str': None
        })
    elif config['analysis_mode'] == 'top-hat-k':
        config.update({
            'dens_case': 'original',
            'dens_cut_str': None,
            'kmax_cut_str': '0.1'
        })
    
    return config

def get_model_parameters():
    """
    Get the optimal hyperparameters found through Optuna optimization.
    
    Returns:
        dict: Dictionary containing the best model parameters
    """
    return {
        'lr': 3.2458049378231884e-05,
        'wd': 3.5921264847791537e-06,
        'channel_1': 20,
        'channel_2': 18,
        'channel_3': 39,
        'channel_4': 179,
        'channel_5': 269
    }

# =============================================================================
# DEVICE AND GPU SETUP
# =============================================================================

def setup_device():
    """
    Configure and setup the computing device (GPU/CPU) for model inference.
    
    Returns:
        torch.device: Configured device for computation
    """
    cudnn.benchmark = True  # May train faster but cost more memory
    
    if torch.cuda.is_available():
        print("Using CUDA")
        device = torch.device('cuda')
        
        if torch.cuda.device_count() > 1:
            print(f"{torch.cuda.device_count()} GPUs Available")
        print(f'GPU model: {torch.cuda.get_device_name()}')
    else:
        print('CUDA Not Available')
        print('Using CPU (Cuda unavailable)')
        device = torch.device('cpu')
    
    return device

# =============================================================================
# DATA LOADING AND PREPARATION
# =============================================================================

def create_test_dataloader(config):
    """
    Create and configure the test dataset and dataloader.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        DataLoader: Configured test dataloader
        data_gen: Test dataset object
    """
    print("Creating test dataset...")
    
    data_set_test = data_gen(
        config['n_samples'], 
        'test', 
        config['dens_case'], 
        config['dens_cut_str'], 
        config['kmax_cut_str'], 
        config['A_true']
    )
    
    test_dl = DataLoader(
        dataset=data_set_test, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    print(f'Size of test dataset = {data_set_test.__len__()}')
    return test_dl, data_set_test

def get_weights_directory(config):
    """
    Determine the directory path where model weights are stored based on analysis mode.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        str: Path to weights directory
    """
    if config['analysis_mode'] == 'original':
        dir_wt = f"/mnt/ceph/users/alahiry/gaussian_fields/{config['analysis_mode']}"
    elif config['analysis_mode'] == 'density':
        dir_wt = f"/mnt/ceph/users/alahiry/gaussian_fields/{config['analysis_mode']}/{config['dens_case']}/{config['dens_cut_str']}"
    elif config['analysis_mode'] == 'top-hat-k':
        dir_wt = f"/mnt/ceph/users/alahiry/gaussian_fields/{config['analysis_mode']}/{config['kmax_cut_str']}"
    
    print(f'Weights folder: {dir_wt}')
    return dir_wt

# =============================================================================
# MODEL LOADING AND SETUP
# =============================================================================

def load_model(params_final, device, dir_wt):
    """
    Load and configure the CNN model with pre-trained weights.
    
    Args:
        params_final (dict): Model hyperparameters
        device (torch.device): Computing device
        dir_wt (str): Directory containing model weights
        
    Returns:
        torch.nn.Module: Loaded model ready for inference
    """
    print("Loading model...")
    
    # Initialize model
    model = Model_CNN_GDF(params_final)
    
    # Setup for multi-GPU if available
    if device == torch.device('cuda') and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.to(device)
    
    # Load pre-trained weights
    fweights_best = os.path.join(dir_wt, 'weights_best.pt')
    
    if os.path.exists(fweights_best):
        model.load_state_dict(torch.load(fweights_best, map_location=device))
        print('Model and weights loaded!')
    else:
        print(f"Warning: Weights file not found at {fweights_best}")
    
    # Display model parameters
    total_params = sum(t.numel() for t in model.parameters())
    print(f'Total number of parameters in the model = {total_params}')
    
    return model

# =============================================================================
# MODEL EVALUATION
# =============================================================================

def evaluate_model(model, test_dl, device):
    """
    Evaluate the model on the test dataset and return predictions vs true values.
    
    Args:
        model (torch.nn.Module): Trained model
        test_dl (DataLoader): Test dataloader
        device (torch.device): Computing device
        
    Returns:
        tuple: (true_values, predictions) both rescaled to original range [0.8, 1.2]
    """
    print("Evaluating model on test set...")
    
    model.eval()
    n_samples = len(test_dl.dataset)
    A_true = np.zeros(n_samples)
    A_nn = np.zeros(n_samples)
    points = 0
    
    with torch.no_grad():
        for images, labels in test_dl:
            images = images.to(device)
            labels = labels.to(device)
            
            # Get model predictions
            out = model(images)
            
            # Store results
            batch_size = images.shape[0]
            A_true[points:points + batch_size] = labels.cpu().numpy().flatten()
            A_nn[points:points + batch_size] = out.cpu().numpy().flatten()
            points += batch_size
    
    # Rescale from [0,1] back to [0.8, 1.2]
    A_nn = A_nn * (1.2 - 0.8) + 0.8
    A_true = A_true * (1.2 - 0.8) + 0.8
    
    return A_true, A_nn

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_predictions_vs_truth(A_true, A_nn, config):
    """
    Create a scatter plot comparing neural network predictions with true values.
    
    Args:
        A_true (np.array): True cosmological parameter values
        A_nn (np.array): Neural network predictions
        config (dict): Configuration parameters for plot labeling
    """
    plt.style.use('seaborn-whitegrid')
    
    # Calculate test loss
    test_loss = mean_squared_error(A_nn, A_true)
    print(f"Test MSE: {test_loss}")
    
    # Determine plot title based on analysis mode
    if config['analysis_mode'] == 'original':
        title = 'Original'
    elif config['analysis_mode'] == 'density':
        if config['dens_case'] == 'max':
            title = r'$\rho_{max} =$' + config['dens_cut_str']
        elif config['dens_case'] == 'min':
            title = r'$\rho_{min} =$' + config['dens_cut_str']
    elif config['analysis_mode'] == 'top-hat-k':
        title = r'$k_{max} =$' + config['kmax_cut_str']
    
    # Create scatter plot
    plt.figure(figsize=(8, 8))
    plt.scatter(A_true, A_nn, s=4, color='xkcd:green', alpha=0.7, 
                label=f'\nNeural Network\nRMSE = {np.sqrt(test_loss):.4f}')
    plt.plot(A_true, A_true, color='black', linewidth=2, label='Truth')
    
    plt.xlabel('$A_{true}$', fontsize=16)
    plt.ylabel('$A_{NN}$', fontsize=16)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.title(title, fontsize=16)
    plt.legend(fontsize=11, frameon=True)
    plt.show()

def create_histogram_datasets(config):
    """
    Create datasets for histogram analysis with fixed A values.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        list: List of dataloaders for different A values
    """
    n = config['test_hist_samples']
    idx = np.random.permutation(n)
    
    # Create datasets for A = 0.82, 1.0, 1.18
    datasets = []
    dataloaders = []
    
    A_values = [0.82, 1.0, 1.18]
    
    for A_val in A_values:
        dataset = data_gen(n, 'all', config['dens_case'], 
                          config['dens_cut_str'], config['kmax_cut_str'], A_val)
        datasets.append(dataset)
        
        sampler = SubsetRandomSampler(idx)
        dataloader = DataLoader(dataset, batch_size=len(idx), sampler=sampler)
        dataloaders.append(dataloader)
    
    return dataloaders, A_values

def analyze_prediction_distributions(model, dataloaders, A_values, device):
    """
    Analyze the distribution of predictions for fixed true A values.
    
    Args:
        model (torch.nn.Module): Trained model
        dataloaders (list): List of dataloaders for different A values
        A_values (list): List of true A values
        device (torch.device): Computing device
        
    Returns:
        tuple: (predictions_array, true_values_array, mse_values, rmse_values)
    """
    n = len(dataloaders[0].dataset)
    A_pred = np.zeros((len(A_values), n))
    A_true = np.zeros((len(A_values), n))
    mse_final = np.zeros(len(A_values))
    rmse_final = np.zeros(len(A_values))
    
    model.eval()
    
    for j, dataloader in enumerate(dataloaders):
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)
                out = model(images)
                
                # Rescale predictions and labels
                for i in range(len(images)):
                    A_pred[j][i] = out[i].item() * (1.2 - 0.8) + 0.8
                    A_true[j][i] = labels[i].item() * (1.2 - 0.8) + 0.8
        
        # Calculate metrics
        mse = mean_squared_error(A_pred[j], A_true[j])
        mse_final[j] = mse
        rmse_final[j] = np.sqrt(mse)
    
    return A_pred, A_true, mse_final, rmse_final

def plot_prediction_histograms(A_pred, A_values):
    """
    Create histogram plots showing the distribution of predictions for fixed A values.
    
    Args:
        A_pred (np.array): Array of predictions for different A values
        A_values (list): List of true A values used
    """
    plt.style.use('seaborn-whitegrid')
    
    fig, ax = plt.subplots(figsize=(25, 10))
    
    colors = ['xkcd:green', 'seagreen', 'lightseagreen']
    labels = [f'$A_{{NN}}$ for $A_{{true}} = {A}$' for A in A_values]
    
    for i, (A_val, color, label) in enumerate(zip(A_values, colors, labels)):
        # Theoretical parameters for Fisher matrix prediction
        mu_th = A_val
        std_th = A_val * np.sqrt(2) / 64
        
        # Plot histogram
        alpha_val = 0.5 + i * 0.1  # Varying transparency
        ax.hist(A_pred[i], bins=100, density=True, color=color, 
                alpha=alpha_val, label=label)
        
        # Plot theoretical Fisher matrix prediction
        x_range = np.linspace(A_val - 0.15, A_val + 0.15, 1000)
        p_fit = norm.pdf(x_range, mu_th, std_th)
        
        if i == 1:  # Add label only once
            ax.plot(x_range, p_fit, color='black', linewidth=3, 
                   linestyle='dashed', label='Fisher Matrix Prediction')
        else:
            ax.plot(x_range, p_fit, color='black', linewidth=3, linestyle='dashed')
        
        # Add text annotations
        y_pos = 43 - i * 8  # Adjust vertical position
        x_pos = A_val - 0.013 if A_val < 1.0 else A_val - 0.04
        
        ax.text(x_pos, y_pos, f"{np.mean(A_pred[i]):.3f} ± {np.std(A_pred[i]):.3f}", 
                fontsize=20, color=color)
        ax.text(x_pos, y_pos - 2, f"{mu_th:.3f} ± {std_th:.3f}", 
                fontsize=20, color='black')
    
    # Configure plot
    ax.xaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    
    plt.ylim(0, 48)
    plt.xlim(0.75, 1.25)
    plt.xticks(fontsize=19)
    plt.yticks(fontsize=19)
    plt.xlabel('\n$A_{NN}$\n', fontsize=25)
    plt.ylabel('\nDensity\n', fontsize=25)
    plt.title('\nHistogram of $A_{NN}$ for fixed values of $A_{true}$ = 0.82, 1.0 and 1.18\n', 
              fontsize=25)
    plt.legend(loc=9, fontsize=22, frameon=True)
    plt.show()

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """
    Main execution function that runs the complete testing pipeline.
    """
    print("=" * 80)
    print("CNN2D-GDF Testing Pipeline")
    print("=" * 80)
    
    # Setup configuration
    config = setup_analysis_parameters()
    params_final = get_model_parameters()
    
    # Setup device
    device = setup_device()
    
    # Create test dataloader
    test_dl, data_set_test = create_test_dataloader(config)
    
    # Get weights directory
    dir_wt = get_weights_directory(config)
    
    # Load model
    model = load_model(params_final, device, dir_wt)
    
    # Evaluate model
    A_true, A_nn = evaluate_model(model, test_dl, device)
    
    # Plot predictions vs truth
    print("\nCreating prediction vs truth plot...")
    plot_predictions_vs_truth(A_true, A_nn, config)
    
    # Histogram analysis
    print("\nPerforming histogram analysis...")
    hist_dataloaders, A_values = create_histogram_datasets(config)
    A_pred, A_true_hist, mse_final, rmse_final = analyze_prediction_distributions(
        model, hist_dataloaders, A_values, device)
    
    # Plot histograms
    print("Creating prediction distribution histograms...")
    plot_prediction_histograms(A_pred, A_values)
    
    print("\nTesting pipeline completed successfully!")
    print("=" * 80)

# Run the main pipeline
if __name__ == "__main__":
    main()