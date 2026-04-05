import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from tqdm.notebook import tqdm

# Set Hyperparameters
hidden_layer_size = 500
learning_rate     = 0.1
training_epochs   = 1

# Get Training Data
mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

# Set up model
## Initialize Model Parameters
W0 = torch.randn(784,hidden_layer_size)/np.sqrt(784)
W0.requires_grad_()
b0 = torch.zeros(hidden_layer_size, requires_grad=True)
W1 = torch.randn(hidden_layer_size,10)/np.sqrt(hidden_layer_size)
W1.requires_grad_()
b1 = torch.zeros(10, requires_grad=True)
## Optimizer
optimizer = torch.optim.SGD([W0,b0,W1,b1], lr=learning_rate)

# Train model on training data
## Iterate through training epochs
for ii in range(training_epochs):
    ## Iterate through training data mini batches
    for images,labels in tqdm(train_loader):
        # zero out the gradients
        optimizer.zero_grad()
    
        # Forward Pass
        x = images.view(-1, 28*28)
        y0 = torch.matmul(x,W0)+b0
        y0_ReLU = F.relu(y0)
        y1 = torch.matmul(y0_ReLU,W1)+b1
        cross_entropy = F.cross_entropy(y1,labels)
        # Backward Pass
        cross_entropy.backward()
        optimizer.step()
    
# Print out accuracy on the test set
correct = 0
total = len(mnist_test)

with torch.no_grad():
    # Iterate through test set mini batches
    for images,labels in tqdm(test_loader):
        # Forward pass
        x = images.view(-1, 28*28)
        y0 = torch.matmul(x,W0)+b0
        y0_ReLU = F.relu(y0)
        y1 = torch.matmul(y0_ReLU,W1)+b1
        
        predictions = torch.argmax(y1,dim=1)
        correct += torch.sum((predictions == labels).float())

print("Test Accuracy: {}".format(correct/total))