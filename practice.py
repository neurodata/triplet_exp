#%%
import torch
import torchvision
from torchvision import transforms, datasets

#%%
x = torch.Tensor([5,3])
y = torch.Tensor([2,1])

print(x*y)
# %%
train = datasets.MNIST("", train=True, download=True,
                transform = transforms.Compose([transforms.ToTensor()]))


test = datasets.MNIST("", train=False, download=True,
                transform = transforms.Compose([transforms.ToTensor()]))
           
# %%
trainset = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=10, shuffle=True)

# %%
for data in trainset:
    print(data)
    break
# %%
x, y = data[0][0], data[1][0]
# %%
import matplotlib.pyplot as plt

plt.imshow(data[0][9].view(28,28))
# %%
import torch.nn as nn
import torch.nn.functional as F 
# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10) 
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return F.log_softmax(x, dim=1)

net = Net()
print(net)   
# %%
X = torch.rand((28,28))
# %%
output = net(X.view(1,784))
# %%
import torch.optim as optim

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 3

for epoch in range(EPOCHS):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1,28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)
# %%
correct = 0
total = 0

with torch.no_grad():
    for data in trainset:
        X, y = data
        output = net(X.view(-1,28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct += 1
            total += 1
print("Accuracy: ", round(correct/total, 3))
# %%
import matplotlib.pyplot as plt 
plt.imshow(X[0].view(28,28))
plt.show()
# %%
print(torch.argmax(net(X[0].view(-1,784))))
# %%
class Net(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_size=10, n_hidden=2,
                activation=torch.nn.ReLU(), bias=False, penultimate=False, bn=False):
        super(Net, self).__init__()

        module = nn.ModuleList()
        module.append(nn.Linear(in_dim, hidden_size, bias=bias))

        for ll in range(n_hidden):
            module.append( activation )
            if bn:
                module.append( nn.BatchNorm1d( hidden_size ) )
            module.append( nn.Linear(hidden_size, hidden_size, bias=bias) )      
        
        if penultimate:
            module.append( activation )
            if bn:
                module.append( nn.BatchNorm1d( hidden_size ) )
            module.append( nn.Linear(hidden_size, 2, bias=bias) )
            hidden_size = 2
            
        module.append( activation )
        if bn:
            module.append( nn.BatchNorm1d( hidden_size ) )
        module.append( nn.Linear(hidden_size, out_dim, bias=bias) )

        self.sequential = nn.Sequential(*module)

    def forward(self, x):
        return self.sequential(x)
# %%
