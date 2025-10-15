import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Model_CNN_GDF(nn.Module):
    """
    A PyTorch neural network model implementing a 2D Convolutional Neural Network (CNN) with
    progressive channel expansion and batch normalization, designed for single-channel input images.
    Args:
        params (dict): Dictionary containing the following keys:
            - 'channel_1': Number of output channels for the first convolutional layer.
            - 'channel_2': Number of output channels for the second convolutional layer.
            - 'channel_3': Number of output channels for the third convolutional layer.
            - 'channel_4': Number of output channels for the fourth convolutional layer.
            - 'channel_5': Number of output channels for the fifth convolutional layer.
    Attributes:
        C01, C02, C03, C04, C05 (nn.Conv2d): Convolutional layers with kernel size 4, stride 2, and padding 1.
        B02, B03, B04, B05 (nn.BatchNorm2d): Batch normalization layers for channels 2 to 5.
        LeakyReLU (nn.LeakyReLU): LeakyReLU activation function with negative slope 0.2.
        Flatten (nn.Flatten): Flattens the output for the fully connected layer.
        FC1 (nn.Linear): Fully connected layer mapping the flattened features to a single output.
    Forward Pass:
        Applies sequential convolution, batch normalization, and LeakyReLU activation to the input image,
        flattens the output, and passes it through a fully connected layer to produce a single output value.
    Example:
        >>> params = {
        ...     'channel_1': 64,
        ...     'channel_2': 128,
        ...     'channel_3': 256,
        ...     'channel_4': 512,
        ...     'channel_5': 1024
        ... }
        >>> model = Model_CNN_GDF(params)
        >>> input_image = torch.randn(1, 1, 64, 64)
        >>> output = model(input_image)
    """


    def __init__(self, params):

        """
        Initializes the Model_CNN_GDF with the specified channel parameters.
        Args:
            params (dict): Dictionary containing channel sizes for each convolutional layer.
        """

        super(Model_CNN_GDF, self).__init__()

        self.C01 = nn.Conv2d(1,params['channel_1'],kernel_size = 4, stride = 2, padding = 1)
        self.C02 = nn.Conv2d(params['channel_1'],params['channel_2'],kernel_size = 4, stride = 2, padding = 1)
        self.C03 = nn.Conv2d(params['channel_2'],params['channel_3'],kernel_size = 4, stride = 2, padding = 1)
        self.C04 = nn.Conv2d(params['channel_3'],params['channel_4'],kernel_size = 4, stride = 2, padding = 1)
        self.C05 = nn.Conv2d(params['channel_4'],params['channel_5'],kernel_size = 4, stride = 2, padding = 1)

        self.B02 = nn.BatchNorm2d(params['channel_2'])
        self.B03 = nn.BatchNorm2d(params['channel_3'])
        self.B04 = nn.BatchNorm2d(params['channel_4'])
        self.B05 = nn.BatchNorm2d(params['channel_5'])

        self.LeakyReLU = nn.LeakyReLU(0.2)

        self.Flatten = nn.Flatten()

        a = params['channel_5']*2*2

        self.FC1 = nn.Linear(a,1)


    def forward(self, image):

        """
        Defines the forward pass of the network.
        Args:
            image (torch.Tensor): Input tensor of shape (batch_size, 1, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """

        out = self.LeakyReLU(self.C01(image))
        out = self.LeakyReLU(self.B02(self.C02(out)))
        out = self.LeakyReLU(self.B03(self.C03(out)))
        out = self.LeakyReLU(self.B04(self.C04(out)))
        out = self.LeakyReLU(self.B05(self.C05(out)))
        out = self.Flatten(out)
        out = self.FC1(out)

        return out