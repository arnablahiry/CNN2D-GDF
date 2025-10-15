"""
CNN2D-GDF Training Pipeline
===========================

This notebook provides a comprehensive training pipeline for a 2D Convolutional Neural Network 
designed to analyze Gaussian Density Fields (GDF). The pipeline includes:

1. Data loading and preprocessing for training and validation
2. Data visualization and exploration
3. GPU/CPU device configuration
4. Hyperparameter optimization using Optuna
5. Model training with cross-validation
6. Model checkpointing and performance tracking

The model learns to predict cosmological parameter A from 2D maps generated from Gaussian random fields.
The training uses Optuna for automated hyperparameter optimization with 50 trials and 200 epochs per trial.
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

# PyTorch and training utilities
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

# =============================================================================
# CONFIGURATION AND PARAMETERS
# =============================================================================

def setup_training_parameters():
    """
    Configure training parameters and analysis modes.
    
    Returns:
        dict: Dictionary containing all configuration parameters for training
    """
    config = {
        'n_samples': 100000,  # Total number of samples in the dataset
        'analysis_mode': 'original',  # Options: 'original', 'density', 'top-hat-k'
        'case': 'original',  # Type of cuts on GDF maps
        'A_true': None,  # Fixed cosmological parameter A (None for uniform sampling [0.8,1.2])
        'batch_size': 128,
        'n_optuna_trials': 51,  # Number of Optuna optimization trials
        'epochs_per_trial': 200,  # Training epochs per trial
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

# =============================================================================
# DEVICE AND GPU SETUP
# =============================================================================

def setup_device():
    """
    Configure and setup the computing device (GPU/CPU) for model training.
    
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

def create_train_val_dataloaders(config):
    """
    Create and configure the training and validation datasets and dataloaders.
    
    Args:
        config (dict): Configuration parameters
        
    Returns:
        tuple: (train_dataloader, validation_dataloader, train_dataset, validation_dataset)
    """
    print("Creating training and validation datasets...")
    
    # Create training dataset
    data_set_train = data_gen(
        config['n_samples'], 
        'train', 
        config['dens_case'], 
        config['dens_cut_str'], 
        config['kmax_cut_str'], 
        config['A_true']
    )
    
    train_dl = DataLoader(
        dataset=data_set_train, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    print(f'Size of train dataset = {data_set_train.__len__()}')
    
    # Create validation dataset
    data_set_valid = data_gen(
        config['n_samples'], 
        'valid', 
        config['dens_case'], 
        config['dens_cut_str'], 
        config['kmax_cut_str'], 
        config['A_true']
    )
    
    valid_dl = DataLoader(
        dataset=data_set_valid, 
        batch_size=config['batch_size'], 
        shuffle=True
    )
    
    print(f'Size of validation dataset = {data_set_valid.__len__()}')
    
    return train_dl, valid_dl, data_set_train, data_set_valid

def get_weights_directory(config):
    """
    Determine and create the directory path where model weights will be stored.
    
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
    
    # Create directory if it doesn't exist
    if not os.path.exists(dir_wt):
        os.makedirs(dir_wt)
        print(f'Created directory: {dir_wt}')
    
    print(f'Weights folder: {dir_wt}')
    return dir_wt

# =============================================================================
# DATA VISUALIZATION
# =============================================================================

def visualize_data_distribution(data_set_valid):
    """
    Visualize the distribution of pixel values across all maps in the validation dataset.
    
    Args:
        data_set_valid: Validation dataset object
    """
    print("Visualizing data distribution...")
    
    # Get all maps and flatten to show pixel value distribution
    all_maps = data_set_valid.full_data()
    all_maps_flat = all_maps.numpy().flatten()
    
    plt.style.use('default')
    plt.style.use('seaborn-whitegrid')
    
    plt.figure(figsize=(8, 5))
    plt.hist(all_maps_flat, bins=70, alpha=0.5, color='xkcd:green', 
             label='Gaussian Density Fields\npixel values')
    plt.yscale('log')
    plt.xlabel('Pixel Values')
    plt.ylabel('Frequency (log scale)')
    plt.title('Distribution of Pixel Values in Gaussian Density Fields')
    plt.legend(frameon=True)
    plt.show()

def visualize_sample_map(valid_dl):
    """
    Visualize a sample Gaussian density field from the validation dataset.
    
    Args:
        valid_dl: Validation dataloader
    """
    print("Visualizing sample Gaussian density field...")
    
    plt.style.use('default')
    
    for images, labels in valid_dl:
        # Select the 10th image from the first batch
        image = images[10][0, :, :]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(image, cmap='viridis', interpolation='bicubic')
        plt.grid(False)
        plt.colorbar(label='Density Value')
        plt.title('Sample Gaussian Density Field Map')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        break
    
    plt.show()

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def fit(params, epochs, model, train_dl, valid_dl, trial, device, dir_wt):
    """
    Train the model for a specified number of epochs and track performance.
    
    Args:
        params (dict): Model hyperparameters
        epochs (int): Number of training epochs
        model (torch.nn.Module): Neural network model
        train_dl (DataLoader): Training dataloader
        valid_dl (DataLoader): Validation dataloader
        trial: Optuna trial object
        device (torch.device): Computing device
        dir_wt (str): Directory to save weights and logs
        
    Returns:
        float: Best validation loss achieved during training
    """
    min_valid_loss = 10**34
    
    lr = params['lr']
    wd = params['wd']
    
    loss_fn = F.mse_loss
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=lr/100, 
        max_lr=lr*100, 
        cycle_momentum=False
    )
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        train_loss1 = torch.zeros(1).to(device)
        train_loss, counts = 0.0, 0
        model.train()
        
        for images, labels in train_dl:
            bs = images.shape[0]
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            preds = model(images)
            loss = loss_fn(preds, labels)
            
            # Accumulate loss
            train_loss1 += loss * bs
            counts += bs
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
        
        train_loss = train_loss1 / counts
        train_loss = torch.mean(train_loss).item()
        
        # Validation phase
        valid_loss1 = torch.zeros(1).to(device)
        valid_loss, counts = 0.0, 0
        model.eval()
        
        with torch.no_grad():
            for images, labels in valid_dl:
                bs = images.shape[0]
                images = images.to(device)
                labels = labels.to(device)
                preds = model(images)
                loss = loss_fn(preds, labels)
                valid_loss1 += loss * bs
                counts += bs
        
        valid_loss = valid_loss1 / counts
        valid_loss = torch.mean(valid_loss).item()
        
        # Save model if validation loss improved
        if valid_loss < min_valid_loss:
            fweights = os.path.join(dir_wt, f'weights_{trial.number}.pt')
            torch.save(model.state_dict(), fweights)
            min_valid_loss = valid_loss
        
        # Log training progress
        floss = os.path.join(dir_wt, f'losses_{trial.number}.txt')
        with open(floss, 'a') as f:
            f.write(f'{epoch} {train_loss:.5e} {valid_loss:.5e}\n')
    
    return min_valid_loss

def objective(trial, config, train_dl, valid_dl, device, dir_wt):
    """
    Optuna objective function for hyperparameter optimization.
    
    Args:
        trial: Optuna trial object
        config (dict): Training configuration parameters
        train_dl (DataLoader): Training dataloader
        valid_dl (DataLoader): Validation dataloader
        device (torch.device): Computing device
        dir_wt (str): Directory for saving weights and logs
        
    Returns:
        float: Best validation loss for this trial
    """
    # Suggest hyperparameters
    params = {
        'lr': trial.suggest_float("lr", 1e-6, 5e-3, log=True),
        'wd': trial.suggest_float('wd', 1e-7, 1e-1, log=True),
        'channel_1': trial.suggest_int("channel_1", 4, 20),
        'channel_2': trial.suggest_int("channel_2", 10, 50),
        'channel_3': trial.suggest_int("channel_3", 20, 80),
        'channel_4': trial.suggest_int("channel_4", 30, 200),
        'channel_5': trial.suggest_int("channel_5", 40, 300)
    }
    
    # Create model with suggested parameters
    model = Model_CNN_GDF(params)
    
    # Setup multi-GPU if available
    if device == torch.device('cuda') and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    model.to(device)
    
    # Train model
    best_loss = fit(
        params, 
        config['epochs_per_trial'], 
        model, 
        train_dl, 
        valid_dl, 
        trial, 
        device, 
        dir_wt
    )
    
    return best_loss

# =============================================================================
# OPTUNA STUDY SETUP AND EXECUTION
# =============================================================================

def setup_optuna_study(config, dir_wt):
    """
    Setup Optuna study for hyperparameter optimization.
    
    Args:
        config (dict): Training configuration
        dir_wt (str): Directory for storing study database
        
    Returns:
        tuple: (study_name, storage_url)
    """
    if config['case'] == 'original':
        study_name = f"AstroNone_GDF_{config['case']}"
        storage = f"sqlite:///{dir_wt}/AstroNone_GDF_{config['case']}.db"
    else:
        study_name = f"AstroNone_GDF_{config['case']}_{config['dens_cut_str']}"
        storage = f"sqlite:///{dir_wt}/AstroNone_GDF_{config['case']}_{config['dens_cut_str']}.db"
    
    return study_name, storage

def run_hyperparameter_optimization(config, train_dl, valid_dl, device, dir_wt):
    """
    Execute the complete hyperparameter optimization using Optuna.
    
    Args:
        config (dict): Training configuration
        train_dl (DataLoader): Training dataloader
        valid_dl (DataLoader): Validation dataloader
        device (torch.device): Computing device
        dir_wt (str): Directory for saving results
        
    Returns:
        optuna.Study: Completed Optuna study object
    """
    print("Starting hyperparameter optimization with Optuna...")
    print(f"Number of trials: {config['n_optuna_trials']}")
    print(f"Epochs per trial: {config['epochs_per_trial']}")
    
    # Setup Optuna study
    study_name, storage = setup_optuna_study(config, dir_wt)
    
    start_time = time.time()
    
    # Create and run study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        study_name=study_name,
        storage=storage,
        load_if_exists=True
    )
    
    # Define objective function with fixed parameters
    def trial_objective(trial):
        return objective(trial, config, train_dl, valid_dl, device, dir_wt)
    
    # Optimize hyperparameters
    study.optimize(trial_objective, n_trials=config['n_optuna_trials'])
    
    end_time = time.time()
    training_time = (end_time - start_time) / 3600.0
    
    print(f'\nHyperparameter optimization completed!')
    print(f'Time taken: {training_time:.4f} hours')
    print(f'Best trial number: {study.best_trial.number}')
    print(f'Best validation loss: {study.best_value:.6e}')
    print(f'Best parameters: {study.best_params}')
    
    return study

# =============================================================================
# MAIN EXECUTION PIPELINE
# =============================================================================

def main():
    """
    Main execution function that runs the complete training pipeline.
    """
    print("=" * 80)
    print("CNN2D-GDF Training Pipeline")
    print("=" * 80)
    
    # Setup configuration
    config = setup_training_parameters()
    print(f"Analysis mode: {config['analysis_mode']}")
    print(f"Dataset size: {config['n_samples']} samples")
    print(f"Batch size: {config['batch_size']}")
    
    # Setup device
    device = setup_device()
    
    # Create dataloaders
    train_dl, valid_dl, data_set_train, data_set_valid = create_train_val_dataloaders(config)
    
    # Setup weights directory
    dir_wt = get_weights_directory(config)
    
    # Visualize data
    print("\nVisualizing training data...")
    visualize_data_distribution(data_set_valid)
    visualize_sample_map(valid_dl)
    
    # Run hyperparameter optimization
    print("\nStarting training with hyperparameter optimization...")
    study = run_hyperparameter_optimization(config, train_dl, valid_dl, device, dir_wt)
    
    # Save best model weights with a consistent name
    best_trial_weights = os.path.join(dir_wt, f'weights_{study.best_trial.number}.pt')
    best_weights_final = os.path.join(dir_wt, 'weights_best.pt')
    
    if os.path.exists(best_trial_weights):
        import shutil
        shutil.copy2(best_trial_weights, best_weights_final)
        print(f"Best model weights saved as: {best_weights_final}")
    
    print("\nTraining pipeline completed successfully!")
    print("=" * 80)
    
    return study

# Run the main pipeline
if __name__ == "__main__":
    study = main()