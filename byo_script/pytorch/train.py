
import argparse
import numpy as np
import os
import sys
import logging
import json
import shutil
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms
from IPython.display import clear_output

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
#import seaborn as sns

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))


def parse_args():
    """
    Parse arguments passed from the SageMaker API
    to the container
    """

    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument("--batch_size", type=int, default=64)


# Data directories
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    # Model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))

    return parser.parse_known_args()

def evaluate_net(net, loader):
    total = 0
    correct = 0
    
    for data in loader:
        inputs, labels = data
        outputs = net(inputs)
        
        total += labels.shape[0]
        correct += (torch.argmax(outputs, dim=1) == labels).float().sum().item()
        
    return correct / total

## model save and load function
def save_model(model, model_dir):
    path = os.path.join(model_dir, "model.pth")
    torch.save(net.state_dict(), path)
    print("saving model to " + model_dir)

def model_fn(model_dir):
    model = Net()
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model
    
    
if __name__ == "__main__":

    args, _ = parse_args()
    
trans = transforms.Compose(
    [transforms.ToTensor()]
)

trainset = datasets.MNIST(args.train, train=True, download=True,transform=trans )
#print(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
#print(trainloader)

testset = datasets.MNIST(args.test, train=True, download=True,transform=trans )
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    

class Net(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
net = Net()    

for i, data in enumerate(trainloader, 0):
    inputs, labels = data
#   print(net(inputs).shape)
#     assert torch.exp(net(inputs)).sum().item() == 100
#     break
    
criterion = nn.NLLLoss()
optimizer = optim.Adam(
    net.parameters(), lr=1e-2
)

loss_history = []


for epoch in range(2):
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        if i % 100 == 100 - 1:
            clear_output(wait=True)
            plt.plot(loss_history)
            plt.show()
            
train_accuracy = evaluate_net(net, trainloader)    
test_accuracy = evaluate_net(net, testloader)         

print(f'Train Accuracy: {train_accuracy}\nTest Accuracy: {test_accuracy}')   

save_model(net, args.model_dir)
