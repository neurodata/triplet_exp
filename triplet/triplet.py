import torch
import torch.nn as nn
import torch.nn.functional as F 

class Net(nn.Module):

    def __init__(self, in_dim, out_dim, width=10, depth=2,
                 activation=torch.nn.ReLU(), bias=False, 
                 penultimate=False):
        super(Net, self).__init__()

        self.layer_number = depth
        module = nn.ModuleList()
        module.append(nn.Linear(in_dim, width, bias=bias))

        for layer in range(depth-1):
            module.append(activation)
            module.append(nn.Linear(width, width, bias=bias))      
        
        if penultimate:
            module.append( activation )
            module.append( nn.Linear(width, out_dim, bias=bias))

        self.sequential = nn.Sequential(*module)

    def freeze_net(self):
        for layer in range(self.layer_number):
            self.sequential.parameters.requires_grad = False

    def forward(self, x):
        return F.softmax(self.sequential(x))


def train_model(model, train_x, train_y, iteration=1000, verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.BCEWithLogitsLoss()
    
    losses = []
        
    for step in range(iteration):
        optimizer.zero_grad()
        outputs = model(train_x)
        loss=loss_func(outputs, train_y)
        trainL = loss.detach().item()

        if verbose and (step % 500 == 0):
            print("train loss = ", trainL)

        losses.append(trainL)
        loss.backward()
        optimizer.step()
    
    return losses