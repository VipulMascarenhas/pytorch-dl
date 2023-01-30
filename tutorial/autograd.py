import torch
from torchvision.models import resnet18, ResNet18_Weights

# torch.autograd is PyTorch’s automatic differentiation engine that
# powers neural network training

# Load a pretrained resnet18 model, create a random data tensor to represent
# a single image with 3 channels and 64x64 height and width, with random labels
# assigned with the shape (1, 1000)
model = resnet18(weights=ResNet18_Weights.DEFAULT)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

# forward pass: run data through the model through each of its layers to make
# a predictions
prediction = model(data)

# calculate error/loss using model's prediction and the corresponding labels.
# Then backpropagate the error through the network using .backward() on the error
# tensor. Autograd calculates and stores the gradient in the parameter's .grad attribute

loss = (prediction - labels).sum()
loss.backward()

# load an optimizer, like SGD, with LR=0.01 and momentum=0.9.
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

# call .step() to initiate gradient descent, where the optimizer adjusts each
# parameter by its gradients stored in .grad
optim.step()

# Autograd
# It keeps a record of tensors and all executed operations in a DAG as objects. In the DAG,
# all the leaves at input tensors and the roots are output tensors.
# By tracing the graph from the roots to the leaves, yop can automatically compute gradients using chain rule.

# Freezing layers
model = resnet18(weights=ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = False

# update the last linear layers model.fc with a new linear layer that is unfrozen by default
model.fc = torch.nn.Linear(512, 10)

# all params of the model except the last one is frozen, so we optimize only the classifier / last layer
optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

