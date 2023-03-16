import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim

batch_size = 200
device = "cuda" if torch.cuda.is_available() else "cpu"
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
test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.main =  nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.main(x)


def test(model, loss_func, dataset):
    dataloader = DataLoader(dataset,batch_size=len(dataset))
    model.eval()
    with torch.no_grad():
        for (X,y) in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            n = len(y)
            loss = loss_func(pred,y).item()/n
            acc = (pred.argmax(dim=1)==y).type(torch.float).mean().item()
            return acc,loss
        
def train_loop(model, dataloader,loss_func, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_func(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 50 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def train(model,loss_func, optimizer, train_dataloader, test_data, epochs=10):
    test_verbose = []
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train_loop(model, train_dataloader,loss_func, optimizer)
        acc,loss = test(model, loss_func, test_data)
        test_verbose.append((acc,loss))
        print(f"Test Error: \n Accuracy: {(100*acc):>0.2f}%, Avg loss: {loss:>10f} \n")
    print("Done!")

if __name__ == "__main__":
    torch.manual_seed(123456)
    net = NeuralNetwork()
    net.to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(),lr=1e-1)
    train(net,loss_func,optimizer,train_loader,test_dataset)
    torch.save(net, 'mnistNet.pth')
