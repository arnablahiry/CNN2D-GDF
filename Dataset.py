import numpy as np
import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import Pk_library as PKL
import density_field_library as DFL
import smoothing_library as SL

class data_gen(Dataset):

    """
    PyTorch Dataset for generating 2D Gaussian density fields with customizable power spectra and preprocessing options.
    Args:
        n (int): Number of samples to generate.
        mode (str): Dataset split mode. One of ['train', 'valid', 'test', 'all'].
        dens_case (str): Density field modification mode. Options: 'original', 'min', or other for max cut.
        dens_cut_str (str or None): Density cut value as string. Used if dens_case is not 'original'.
        kmax_cut_str (str or None): Maximum k value for top-hat filter in Fourier space as string.
        A_true (float or None): True amplitude value for the power spectrum. If None, random amplitude is used per sample.
    Attributes:
        data_t (torch.Tensor): Tensor containing the processed density field data.
        A (torch.Tensor): Tensor containing the normalized amplitude values for each sample.
        size (int): Number of samples in the current dataset split.
    Methods:
        __len__(): Returns the number of samples in the dataset.
        __getitem__(idx): Returns the density field and amplitude for the given index.
        full_data(): Returns the full tensor of density fields.
    Notes:
        - The density fields are generated using a custom power spectrum P(k) ~ A/sqrt(k).
        - Optional top-hat filtering in Fourier space can be applied.
        - Density values can be clipped according to dens_case and dens_cut_str.
        - Data is normalized using fixed mean and standard deviation.
        - Amplitude values are normalized to [0, 1] range.
        - The dataset is split according to the mode argument.
    """

    def __init__(self, n, mode, dens_case, dens_cut_str, kmax_cut_str, A_true):
        super().__init__()

        data = np.zeros((n,64,64))
        A = np.zeros(n)

        seed_arr = np.zeros(n)
        for i in range(n):
            seed_arr[i] = i

        for i in range(n):

            grid              = 64                     
            BoxSize           = 1000           
            seed              = int(seed_arr[i])      
            Rayleigh_sampling = 1           
            threads           = 1                                          
            verbose           = False
            MAS               = 'None'                          
                
            kf = 7e-03
            kmax = 0.9
            k = np.arange(3*kf, kmax, kf)
            k = k.astype(np.float32)

            Pk = []
            if A_true == None:
                np.random.seed(seed)
                A_1 = np.random.uniform(0.8,1.2)
            else:
                A_1 = A_true

            for j in k:
                Pk_1 = A_1/(np.sqrt(j))
                Pk.append(Pk_1)

            Pk = np.array(Pk)
            Pk = Pk.astype(np.float32)

            data_1 = DFL.gaussian_field_2D(grid, k, Pk, Rayleigh_sampling, seed,
                BoxSize, threads, verbose)

            if kmax_cut_str != None:
                filter            = 'Top-Hat-k'
                R                 = 0.0
                k_min             = 0  
                k_max             = float(kmax_cut_str)
                W_k = SL.FT_filter_2D(BoxSize, R, grid, filter, threads, k_min, k_max)
                field_smoothed = SL.field_smoothing_2D(data_1, W_k, threads)
                data[i,:,:] = field_smoothed
            else:
                data[i,:,:] = data_1

            #normalising A wrt maximum and minimum
            A_2 = (A_1 - 0.8)/(1.2-0.8)
            A[i] = A_2

        if dens_case != 'original':
            dens_cut = float(dens_cut_str)
            if dens_case == 'min':
                indexes = np.where(data<dens_cut)
                data[indexes] = dens_cut
            else:
                indexes = np.where(data>dens_cut)
                data[indexes] = dens_cut  

                
        if   mode=='train':  offset, size_maps = int(0.00*n), int(0.70*n)
        elif mode=='valid':  offset, size_maps = int(0.70*n), int(0.15*n)
        elif mode=='test':   offset, size_maps = int(0.85*n), int(0.15*n)
        elif mode=='all':    offset, size_maps = int(0.00*n), int(1.00*n)
        else:                raise Exception('Wrong name!')

        data = data[offset:offset+size_maps,:]
        A = A[offset:offset+size_maps]
        
        #mean and standard deviation of the training set of analysis_mode = 'original'
        #(no augmentations) maps
        #will change depending on the type of data being trained on
        mean, std = 4.864242144507703e-13, 0.10662432912867778
        data = (data - mean)/std
        data = np.expand_dims(data, axis=1)
        data_t = torch.from_numpy(data)
        self.data_t = data_t

        A_t = torch.from_numpy(A)
        A_t = A_t.view(size_maps,1)
        self.A = A_t

        self.size = self.data_t.shape[0]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data_t[idx].to(torch.float32), self.A[idx].to(torch.float32)
    
    def full_data(self):
        return(self.data_t)
    
