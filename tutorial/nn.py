import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

'''Train a ConvNet, which is a simple feed-forward network.
Typical procedure includes the following steps
1. define the NN with some weights
2. iterate over a dataset of inputs
3. process inputs through the network
4. compute the loss
5. propagate the gradients back to the network parameters
6. update the weight of the network
'''


class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolutions
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    # only forward function needs to be defined, backward is auto defined using autograd


net = NeuralNet()
print(net)
'''
NeuralNet(
  (conv1): Conv2d(1, 6, kernel_size=(5, 5), stride=(1, 1))
  (conv2): Conv2d(6, 15, kernel_size=(5, 5), stride=(1, 1))
  (fc1): Linear(in_features=400, out_features=12, bias=True)
  (fc2): Linear(in_features=120, out_features=84, bias=True)
  (fc3): Linear(in_features=84, out_features=10, bias=True)
)
'''

params = list(net.parameters())
print(len(params))
print(params[0].size())
'''
10
torch.Size([6, 1, 5, 5])
'''

# expected input size of this NN is 32x32. For MNIST, the images need to be resized to 32x32
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))

# Computing Loss: loss function takes output, input and computes how far away the output is from the target
output = net(input)
target = torch.randn(10)
# reshape as same as output
target = target.view(1, -1)
# can choose anyone based on the type of problem
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

# Backprop: call loss.backward(), but clear the existing gradients through, else gradients
# will be accumulated to existing gradients

net.zero_grad()
print(f"conv1.bias.grad before backward: \n {net.conv1.bias.grad}")
loss.backward()
print(f"conv1.bias.grad after backward: \n {net.conv1.bias.grad}")

# Last step is to update the weights. Use the rules for Stochastic Gradient Descent
# weight = weight - learning_rate * gradient

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# alternatively, use torch.optim that has various update rules for SGD, Adam, RMSProp, Nesterov-SGD, etc.

# create optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)
# during training
optimizer.zero_grad()
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()

