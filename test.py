import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

"""
Note: expected input size of this net (LeNet) is 32x32. To use this net on the MNIST dataset, please resize the images from the dataset to 32x32.

TUTORIAL : https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
"""

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        """
        1 input image channel, 6 output channels, 3x3 square convolution
        kernel
        Applies a 2D convolution over an input signal composed of several input planes.

        Conv2d:
        https://pytorch.org/docs/master/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        # Se suma la multiplicación de el input-j con el peso del kernel para j. Eso para cada elemento.
        # explicacion: https://medium.com/apache-mxnet/multi-channel-convolutions-explained-with-ms-excel-9bbf8eb77108

        Linear:
        https://pytorch.org/docs/master/generated/torch.nn.Linear.html#torch.nn.Linear
        """
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation (linear transformation): y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Max pooling:
        over a (2, 2) window
        https://computersciencewiki.org/index.php/Max-pooling_/_Pooling
        # Devuelve el maximo en la ventana de 2x2
        # El tamaño de la matriz se divide por 2.

        RELU: ReLU(x)=max(0,x)
        https://pytorch.org/docs/master/generated/torch.nn.ReLU.html#torch.nn.ReLU
        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # pasamos de n dimensiones a un vector
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# instance of network
net = Net()
# Simula una imagen de 32x32 dado que estamos usando el dataset de MNIST. Que tiene 10 clases para clasificar imagenes.
input1 = torch.randn(1, 1, 32, 32)

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.MSELoss()

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input1)

# Genero los valores para 10 clases a comparar con el output
target = torch.randn(10)  # a dummy target, for example
# Transformo target a un vector de una dimension.
target = target.view(1, -1)  # make it the same shape as output

loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update


#https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
# proximo tutorial
# https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html