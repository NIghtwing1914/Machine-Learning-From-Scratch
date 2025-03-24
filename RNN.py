import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim import Adam

lr=0.01
input_size = 28
sequence_length=28
num_layers=2
hidden_size=256
num_classes=10
batch_size=64
epochs = 2

class RNN(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers,num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size,hidden_size,num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size*sequence_length,num_classes)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers,x.size(0),self.hidden_size)

        out, _ = self.rnn(x,h0)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)

model = RNN(input_size,hidden_size,num_layers,num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(),lr=lr)


train_dataset = datasets.MNIST(root = 'dataset/',train=True,transform = transforms.ToTensor(),download=True)   
test_dataset = datasets.MNIST(root = 'dataset/',train=True,transform = transforms.ToTensor(),download=True)

trainloader = DataLoader(dataset = train_dataset,batch_size= batch_size,shuffle=True)
testloader = DataLoader(dataset = test_dataset,batch_size= batch_size,shuffle=True)

for i in range(epochs):
    for batch_idx, (data,targets) in enumerate(trainloader):
        data = data.squeeze(1)
        scores = model(data)
        loss=criterion(scores,targets)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

model.eval()

with torch.no_grad():

   for x,y in trainloader:
       x=x.squeeze(1)
       scores=model(x)

       _,predictions = scores.max(dim=1)
       print(predictions)
