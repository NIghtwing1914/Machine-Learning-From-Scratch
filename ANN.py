import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam

class ANN(nn.Module):
    
    def __init__(self,n_input, n_hidden,n_output):
        super().__init__()
        self.fc1 = nn.Linear(n_input,n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_output)

    def forward(self,x):
        return self.fc2(F.relu(self.fc1(x)))
    

class CNN(nn.Module):

    def __init__(self, in_channels=1,num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc1 = nn.Linear(16*7*7,num_classes)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc1(x)

        return x
    
model = CNN()


lr=0.01
batch_size=32
model =  ANN(784,4,10)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(),lr)
epochs = 10


train_dataset = datasets.MNIST(root = 'dataset/',train=True,transform = transforms.ToTensor(),download=True)   
test_dataset = datasets.MNIST(root = 'dataset/',train=True,transform = transforms.ToTensor(),download=True)

trainloader = DataLoader(dataset = train_dataset,batch_size= batch_size,shuffle=True)
testloader = DataLoader(dataset = test_dataset,batch_size= batch_size,shuffle=True)




for i in range(epochs):
    for batch_idx, (data,targets) in enumerate(trainloader):
        data = data.reshape(data.shape[0],-1)
        scores = model(data)
        loss=criterion(scores,targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

model.eval()

with torch.no_grad():

   for x,y in trainloader:
       x=x.reshape(x.shape[0],-1)
       scores=model(x)

       _,predictions = scores.max(dim=1)
       print(predictions)
