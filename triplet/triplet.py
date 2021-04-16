import torch
import torch.nn as nn
import torch.nn.functional as F 

class Net(nn.Module):

    def __init__(self, in_dim, out_dim, width=10, depth=2,
                 activation=torch.nn.ReLU(), bias=False):
        super(Net, self).__init__()

        self.layer_number = depth
        self.bias = bias

        module = nn.ModuleList()
        module.append(nn.Linear(in_dim, width, bias=bias))

        for layer in range(depth-1):
            module.append(activation)
            module.append(nn.Linear(width, width, bias=bias))      
        
        module.append(activation)
        module.append(nn.Linear(width, out_dim, bias=bias))

        self.sequential = nn.Sequential(*module)

    def freeze_net(self):
        for layer in range(0,self.layer_number,2):
            self.sequential[layer].weight.requires_grad = False

            if self.bias:
                self.sequential[layer].bias.requires_grad = False

    def forward(self, x):
        return self.sequential(x)


def train_model(model, train_x, train_y, iteration=1000, freeze=False, verbose=False):

    if freeze:
        model.freeze_net()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_func = torch.nn.BCEWithLogitsLoss()
    
    losses = []
        
    for step in range(iteration):
        optimizer.zero_grad()
        outputs = model(train_x)
        train_y = train_y.type_as(outputs)
        loss=loss_func(outputs, train_y)
        trainL = loss.detach().item()

        if verbose and (step % 500 == 0):
            print("train loss = ", trainL)

        losses.append(trainL)
        loss.backward()
        optimizer.step()
    
    return losses

def predict(model, X):
    '''return torch.argmax(
        torch.round(torch.sigmoid(model(X))),
        dim=1
    )'''
    return torch.round(torch.sigmoid(model(X)))