import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

batch_size = 200
train_dataset = datasets.MNIST('./data', train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ]))
test_dataset = datasets.MNIST('./data', train=False, download=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.1307,), (0.3081,))
                              ]))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_load = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.main(x)


def test(model, dataset, loss_func):
    dataloader = DataLoader(dataset,batch_size=len(dataset))
    model.eval()
    with torch.no_grad():
        for (X,y) in dataloader:
            pred = model(X)
            n = len(y)
            loss = loss_func(pred,y).item()/n
            acc = (pred.argmax(dim=1)==y).type(torch.float).mean().item()
            return acc,loss
        
def train_loop(model, dataloader, loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        loss = loss_func(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def train(model,loss_func, optimizer, train_dataloader, test_data):
    test_verbose = []
    epochs = 7
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(model, train_dataloader,loss_func, optimizer)
        correct,loss = test(model, test_data, loss_func)
        test_verbose.append((correct,loss))
        print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {loss:>10f} \n")
    print("Done!")


device = "cuda" if torch.cuda.is_available() else "cpu"

torch.random.seed(123456)
net = NeuralNetwork()
loss_func = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=1e-1)
train(net,loss_func,optimizer,train_loader,test_dataset)
torch.save(net, 'mnistNet.pth')
