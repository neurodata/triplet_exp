#%%
from triplet.dd import *
import pandas as pd
import torch
import copy
# %%
reps = 200
sample_size = np.logspace(
        np.log10(2),
        np.log10(1e4),
        num=10,
        endpoint=True,
        dtype=int
        )
test_size = 1000
angle = 30
#%%
np.random.seed(10)
torch.manual_seed(10)
summary = pd.DataFrame()
err_rxor_on_xor1 = []
err_rxor_on_xor2 = []
rep_list = []
sample_list = []
#depth_list = []
#depths = [3,19]
#widths = [3,20]
train_xor_X, train_xor_y, test_xor_X, test_xor_y = get_dataset(N=1000, cov_scale=1)
net_xor1 = get_model(n_hidden = 3, hidden_size = 3, penultimate=False, bn=False) 
net_xor2 = get_model(n_hidden = 19, hidden_size=20, penultimate=False, bn=False)
'''l1 = train_model(
                net_xor1, 
                train_xor_X, 
                train_xor_y,
                iteration=1000
                )
l2 = train_model(
                net_xor2, 
                train_xor_X, 
                train_xor_y,
                iteration=1000
                )'''
model_state_dic = torch.load('saved_model/model_weights_xor_depth_3_.pth',map_location=torch.device('cpu'))
net_xor1.load_state_dict(model_state_dic)

model_state_dic = torch.load('saved_model/model_weights_xor_depth_19_.pth',map_location=torch.device('cpu'))
net_xor2.load_state_dict(model_state_dic)
#%%
for sample in sample_size:
    for ii in range(reps):
        print('doing sample %d rep %d'%(sample,ii+1))
        
        net_xor1_ = copy.deepcopy(net_xor1)
        net_xor2_ = copy.deepcopy(net_xor2)
        #target task
        train_rxor_X, train_rxor_y, test_rxor_X, test_rxor_y = get_dataset(
                N=sample, 
                angle_param=angle,
                cov_scale=1
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
summary.to_csv('triplet_cov_scale1_'+str(200)+'_'+str(len(sample_size))+'_'+str(angle)+'.csv')
# %%
