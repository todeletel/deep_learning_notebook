from torchvision import datasets
import torchvision.transforms as transforms
from torch.nn import functional as F
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 64

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# get the training datasets
train_data = datasets.MNIST(root='data', train=True,
                                   download=True, transform=transform)

# prepare data loader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                           num_workers=num_workers)


# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# get one image from the batch
img = np.squeeze(images[0])

fig = plt.figure(figsize = (3,3)) 
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')
plt.show()



class Discriminator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Discriminator, self).__init__()
        
        # define all layers
        self.hidden_layer = nn.Linear(input_size, hidden_dim)
        
        self.output_layer = nn.Linear(hidden_dim, output_size)
        
    def forward(self, x):
        # flatten image
        hidden_node = self.hidden_layer(x)
        hidden_out = F.leaky_relu(hidden_node)
        output = F.sigmoid(self.output_layer(hidden_out))
        # pass x through all layers
        # apply leaky relu activation to all hidden layers
        return output


class Generator(nn.Module):

    def __init__(self, input_size, hidden_dim, output_size):
        super(Generator, self).__init__()
                # define all layers
        self.hidden_layer = nn.Linear(input_size, hidden_dim)
        
        self.output_layer = nn.Linear(hidden_dim, output_size)
        # define all layers
        

    def forward(self, x):
        # pass x through all layers
        
        # final layer should have tanh applied
        hidden_node = self.hidden_layer(x)
        hidden_out = F.leaky_relu(hidden_node)
        output = F.tanh(self.output_layer(hidden_out))
        return x

Generator().parameters