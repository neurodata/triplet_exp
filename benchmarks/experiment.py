#%%
import torch
import torch.nn.functional as F
from triplet import Net, train_model, predict
from triplet.utils import generate_gaussian_parity
import numpy as np
import pandas as pd
# %%
reps = 10
sample_size = np.logspace(
        np.log10(1e2),
        np.log10(1e4),
        num=10,
        endpoint=True,
        dtype=int
        )
test_size = 1000
xor_train_size = 1000
xor_test_size = 1000
#%%
summary = pd.DataFrame()
err_xor_on_xor = []
err_rxor_on_xor = []
err_rxor_on_rxor = []
rep_list = []
sample_list = []
widths = [5,10]
depths = [5,20]

for width, depth in zip(widths,depths):
    for sample in sample_size:
        for ii in range(reps):
            print('doing sample %d rep %d'%(sample,ii+1))
            net_xor = Net(in_dim=2, out_dim=2, width=width, depth=depth, bias=True)
            net_rxor = Net(in_dim=2, out_dim=2, width=width, depth=depth, bias=True)

            #source task
            train_xor_X, train_xor_y = generate_gaussian_parity(xor_train_size,cluster_std=0.25) 
            train_xor_X, train_xor_y = torch.FloatTensor(train_xor_X), (torch.FloatTensor(train_xor_y).unsqueeze(-1))
            train_xor_X, train_xor_y = generate_gaussian_parity(xor_test_size,cluster_std=0.25)
            test_xor_X, test_xor_y = torch.FloatTensor(test_xor_X), test_xor_y
            #target task
            train_rxor_X, train_rxor_y = generate_gaussian_parity(
                                    sample,
                                    cluster_std=0.25, 
                                    angle_params=np.pi*10/180
                                )
            test_rxor_X, test_rxor_y = generate_gaussian_parity(
                                    test_size, 
                                    cluster_std=0.25,
                                    angle_params=np.pi*10/180
                                )
            train_rxor_X, train_rxor_y = torch.FloatTensor(train_rxor_X), (torch.FloatTensor(train_rxor_y).unsqueeze(-1))
            test_rxor_X, test_rxor_y = torch.FloatTensor(test_rxor_X), test_rxor_y

            train_model(
                net_xor, 
                train_xor_X, 
                F.one_hot(train_xor_y[:,0].to(torch.long)),
                iteration=10000,
                verbose=False
                )
            predicted_label = predict(net_xor, test_xor_X)
            err_xor_on_xor.append(
                1 - np.mean(predicted_label.numpy()==test_xor_y)
            ) 

            train_model(
                net_xor, 
                train_rxor_X, 
                F.one_hot(train_rxor_y[:,0].to(torch.long)), 
                iteration=1000,
                freeze=True, 
                verbose=False
                )

            predicted_label = predict(net_xor, test_rxor_X)
            err_rxor_on_xor.append(
                1 - np.mean(predicted_label.numpy()==test_rxor_y)
            ) 

            train_model(
                net_rxor, 
                train_rxor_X,
                F.one_hot(train_rxor_y[:,0].to(torch.long)), 
                iteration=10000,
                verbose=False
                )
            predicted_label = predict(net_rxor, test_rxor_X)
            err_rxor_on_rxor.append(
                1 - np.mean(predicted_label.numpy()==test_rxor_y)
            )
            
            rep_list.append(ii+1)
            sample_list.append(sample)

summary['error xor on xor transformer'] = err_xor_on_xor
summary['error rxor on xor transformer'] = err_rxor_on_xor
summary['error rxor on rxor transformer'] = err_rxor_on_rxor
summary['rep'] = rep_list
summary['sample'] = sample_list
summary.to_csv('./benchmarks/triplet_with_softmax.csv')
# %%
'''import seaborn as sns
import matplotlib.pyplot as plt 

D_x, D_y = generate_gaussian_parity(1000,cluster_std=.25, angle_params=0)
colors = sns.color_palette('Dark2', n_colors=2)

fig, ax = plt.subplots(1,1, figsize=(8,8))

clr = [colors[i] for i in D_y]
ax.scatter(D_x[:, 0], D_x[:,1], c=clr, s=50)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title('Gaussian XOR', fontsize=30)
ax.axis('off')'''
# %%
