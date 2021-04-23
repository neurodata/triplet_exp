import math
import numpy as np
import torch
from torch import nn
import copy
import random
import concurrent.futures

## Distributions 

def generate_gaussian_parity(n, cov_scale=1, angle_params=None, k=1, acorn=None):
    """ Generate Gaussian XOR, a mixture of four Gaussians elonging to two classes. 
    Class 0 consists of negative samples drawn from two Gaussians with means (−1,−1) and (1,1)
    Class 1 comprises positive samples drawn from the other Gaussians with means (1,−1) and (−1,1) 
    """
#     means = [[-1.5, -1.5], [1.5, 1.5], [1.5, -1.5], [-1.5, 1.5]]
    means = [[-1, -1], [1, 1], [1, -1], [-1, 1]]
    blob = np.concatenate(
        [
            np.random.multivariate_normal(
                mean, cov_scale * np.eye(len(mean)), size=int(n / 4)
            )
            for mean in means
        ]
    )

    X = np.zeros_like(blob)
    Y = np.concatenate([np.ones((int(n / 4))) * int(i < 2) for i in range(len(means))])
    X[:, 0] = blob[:, 0] * np.cos(angle_params * np.pi / 180) + blob[:, 1] * np.sin(
        angle_params * np.pi / 180
    )
    X[:, 1] = -blob[:, 0] * np.sin(angle_params * np.pi / 180) + blob[:, 1] * np.cos(
        angle_params * np.pi / 180
    )
    return X, Y.astype(int)


## Network functions

# Model 
class Net(nn.Module):
    """ DeepNet class
    A deep net architecture with `n_hidden` layers, 
    each having `hidden_size` nodes.
    """
    def __init__(self, in_dim, out_dim, hidden_size=10, n_hidden=2,
                activation=torch.nn.ReLU(), bias=False, penultimate=False, bn=False):
        super(Net, self).__init__()

        module = nn.ModuleList()
        module.append(nn.Linear(in_dim, hidden_size, bias=bias))
        self.layer = 1
        self.bias = bias
        
        for ll in range(n_hidden):
            module.append( activation )
            self.layer += 1
            if bn:
                module.append( nn.BatchNorm1d( hidden_size ) )
            module.append( nn.Linear(hidden_size, hidden_size, bias=bias) )  
            self.layer += 1    
        
        if penultimate:
            module.append( activation )
            self.layer += 1
            if bn:
                module.append( nn.BatchNorm1d( hidden_size ) )
            module.append( nn.Linear(hidden_size, 2, bias=bias) )
            self.layer += 1
            hidden_size = 2
            
        module.append( activation )
        self.layer += 1
        if bn:
            module.append( nn.BatchNorm1d( hidden_size ) )
        module.append( nn.Linear(hidden_size, out_dim, bias=bias) )
        self.layer += 1

        self.sequential = nn.Sequential(*module)

    def freeze_net(self):
        for layer in range(0,self.layer-1,2):
            self.sequential[layer].weight.requires_grad = False

            if self.bias:
                self.sequential[layer].bias.requires_grad = False

    def forward(self, x):
        return self.sequential(x)


def train_model(model, train_x, train_y, multi_label=False, verbose=False):
    """ 
     Train the model given the training data
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_func = torch.nn.BCEWithLogitsLoss()
    
    losses = []
        
    for step in range(1000):
        optimizer.zero_grad()
        outputs = model(train_x)
        if multi_label:
            train_y = train_y.type_as(outputs)
        
        loss=loss_func(outputs, train_y)
        trainL = loss.detach().item()
        if verbose and (step % 500 == 0):
            print("train loss = ", trainL)
        losses.append(trainL)
        loss.backward()
        optimizer.step()
    
    return losses

def get_model(hidden_size=20, n_hidden=5, in_dim=2, out_dim=1, penultimate=False, use_cuda=False, bn=False):
    """
     Initialize the model and send to gpu
    """
    in_dim = in_dim
    out_dim = out_dim #1
    model = Net(in_dim, out_dim, n_hidden=n_hidden, hidden_size=hidden_size,
                activation=torch.nn.ReLU(), bias=True, penultimate=penultimate, bn=bn)
    
    if use_cuda:
        model=model.cuda()
        
    return model

def get_dataset(N=1000, angle_param=0, one_hot=False, cov_scale=1, include_hybrid=False):
    """
     Generate the Gaussian XOR dataset and move to gpu
    """
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(0)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        
    if include_hybrid:
        D_x, D_y = generate_gaussian_parity(cov_scale=cov_scale, n=2*N, angle_params=angle_param)
        D_perm = np.random.permutation(2*N)
        D_x, D_y  = D_x[D_perm,:], D_y[D_perm]
        train_x, train_y = D_x[:N], D_y[:N]
        ghost_x, ghost_y = D_x[N:], D_y[N:]
        hybrid_sets = []
        rand_idx = random.sample(range(0,N-1), N//10)
        for rand_i in rand_idx:
            hybrid_x, hybrid_y = np.copy(train_x), np.copy(train_y)
            hybrid_x[rand_i], hybrid_y[rand_i] = ghost_x[rand_i], ghost_y[rand_i]
            hybrid_x = torch.FloatTensor(hybrid_x)
            hybrid_y = (torch.FloatTensor(hybrid_y).unsqueeze(-1))
            hybrid_x, hybrid_y = hybrid_x.cuda(), hybrid_y.cuda()
            hybrid_sets.append((hybrid_x, hybrid_y))
    else:
        train_x, train_y = generate_gaussian_parity(cov_scale=cov_scale, n=N, angle_params=angle_param)
        train_perm = np.random.permutation(N)
        train_x, train_y = train_x[train_perm,:], train_y[train_perm] 
    test_x, test_y = generate_gaussian_parity(cov_scale=cov_scale, n=1000, angle_params=angle_param)
    
    test_perm = np.random.permutation(1000)
    test_x, test_y  = test_x[test_perm,:], test_y[test_perm]
    
    train_x = torch.FloatTensor(train_x)
    test_x = torch.FloatTensor(test_x)

    train_y = (torch.FloatTensor(train_y).unsqueeze(-1))#[:,0]
    test_y = (torch.FloatTensor(test_y).unsqueeze(-1))#[:,0]
    
    if one_hot:
        train_y = torch.nn.functional.one_hot(train_y[:,0].to(torch.long))
        test_y = torch.nn.functional.one_hot(test_y[:,0].to(torch.long))
    
    # move to gpu
    if use_cuda:
        train_x, train_y = train_x.cuda(), train_y.cuda()
        test_x, test_y = test_x.cuda(), test_y.cuda()
        
    if include_hybrid:
        return train_x, train_y, test_x, test_y, hybrid_sets
    
    return train_x, train_y, test_x, test_y



