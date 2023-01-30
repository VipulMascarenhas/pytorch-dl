import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

'''
Train an image classifier to identify objects in CIFAR-10 dataset.
Steps to do so:
1. Load and normalize the train and test CIFAR-10 data (3x32x32) using torchvision
2. Define a CNN
3. Define a loss function
4. Train the network on training data
5. Test on test data  
'''
# The output of torchvision datasets are PILImage images of range [0, 1].
# Transform them to Tensors of normalized range [-1, 1].


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def prepare_data():

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    batch_size = 4

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes


def train():
    train_loader, test_loader, classes = prepare_data()
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # data is a list of inputs, labels
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:
                print(f"[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}")
                running_loss = 0.0

    print("Finished Training")

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    net = Net()
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    # no need to calculate gradients for outputs since we're not training
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            # class with highest energy is chosen as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy of the network on the 10k test images :{100 * correct // total}%")


if __name__ == "__main__":
    train()

'''
Files already downloaded and verified
Files already downloaded and verified
[1,  2000] loss: 2.219
[1,  4000] loss: 1.839
[1,  6000] loss: 1.662
[1,  8000] loss: 1.567
[1, 10000] loss: 1.512
[1, 12000] loss: 1.469
[2,  2000] loss: 1.389
[2,  4000] loss: 1.368
[2,  6000] loss: 1.327
[2,  8000] loss: 1.300
[2, 10000] loss: 1.291
[2, 12000] loss: 1.291
Finished Training
Accuracy of the network on the 10k test images :56%
'''