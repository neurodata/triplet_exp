#%%
from triplet.dd import *
import pandas as pd
import torch
import copy
# %%
reps = 100
sample_size = np.logspace(
        np.log10(1e1),
        np.log10(1e3),
        num=5,
        endpoint=True,
        dtype=int
        )
test_size = 1000
#%%
#np.random.seed(12345)
summary = pd.DataFrame()
err_rxor_on_xor1 = []
err_rxor_on_xor2 = []
rep_list = []
sample_list = []
#depth_list = []
#depths = [3,19]
#widths = [3,20]
train_xor_X, train_xor_y, test_xor_X, test_xor_y = get_dataset(N=1000, cov_scale=.25)
net_xor1 = get_model(n_hidden = 5, hidden_size = 5, penultimate=False, bn=False) 
net_xor2 = get_model(n_hidden = 15, hidden_size=15, penultimate=False, bn=False)
_ = train_model(
                net_xor1, 
                train_xor_X, 
                train_xor_y,
                )
_ = train_model(
                net_xor2, 
                train_xor_X, 
                train_xor_y,
                )

#%%
for sample in sample_size:
    for ii in range(reps):
        print('doing sample %d rep %d'%(sample,ii+1))
        
        net_xor1_ = copy.deepcopy(net_xor1)
        net_xor2_ = copy.deepcopy(net_xor2)
        #target task
        train_rxor_X, train_rxor_y, test_rxor_X, test_rxor_y = get_dataset(
                N=sample, 
                angle_param=10,
                cov_scale=.25
                )

        net_xor1_.freeze_net()
        net_xor2_.freeze_net()
        train_model(
            net_xor1_, 
            train_rxor_X, 
            train_rxor_y,
            lr=.01,
            iteration=1000
            )
        train_model(
            net_xor2_, 
            train_rxor_X, 
            train_rxor_y,
            lr=0.01,
            iteration=1000
            )

        predicted_label = net_xor1_(test_rxor_X)
        err_rxor_on_xor1.append(
            1 - (torch.sigmoid(predicted_label).round() == test_rxor_y).sum().cpu().data.numpy().item() / test_rxor_y.size(0)
        )
            
        predicted_label = net_xor2_(test_rxor_X)
        err_rxor_on_xor2.append(
            1 - (torch.sigmoid(predicted_label).round() == test_rxor_y).sum().cpu().data.numpy().item() / test_rxor_y.size(0)  
        )
        rep_list.append(ii+1)
        sample_list.append(sample)

summary['error rxor on xor transformer 1'] = err_rxor_on_xor1
summary['error rxor on xor transformer 2'] = err_rxor_on_xor2
summary['rep'] = rep_list
summary['sample'] = sample_list
summary.to_csv('triplet_kalaeb_100_5.csv')
# %%
