#%%
from triplet.dd import *
import pandas as pd
import torch
# %%
reps = 20
sample_size = np.logspace(
        np.log10(1e2),
        np.log10(1e4),
        num=10,
        endpoint=True,
        dtype=int
        )
test_size = 1000
#%%
summary = pd.DataFrame()
err_xor_on_xor = []
err_rxor_on_xor = []
err_rxor_on_rxor = []
rep_list = []
sample_list = []
depth_list = []
depths = [3,19]
widths = [3,20]
train_xor_X, train_xor_y, test_xor_X, test_xor_y = get_dataset(N=1000, cov_scale=1)

for depth,width in zip(depths,widths):
    for sample in sample_size:
        for ii in range(reps):
            print('doing sample %d rep %d'%(sample,ii+1))
            net_xor = get_model(n_hidden = depth, hidden_size=width, penultimate=False, bn=False) 
            net_rxor = get_model(n_hidden = depth, hidden_size=width, penultimate=False, bn=False)
            
            #model_state_dic = torch.load('saved_model/model_weights_xor_depth_'+str(depth-1)+'_.pth',map_location=torch.device('cpu'))
            #net_xor.load_state_dict(model_state_dic)
            
            #target task
            train_rxor_X, train_rxor_y, test_rxor_X, test_rxor_y = get_dataset(
                N=sample, 
                angle_param=10,
                cov_scale=1
                )

            _ = train_model(
                net_xor, 
                train_xor_X, 
                train_xor_y,
                )
            predicted_label = net_xor(test_xor_X)
            err_xor_on_xor.append(
                1 - (torch.sigmoid(predicted_label).round() == test_xor_y).sum().cpu().data.numpy().item() / test_xor_y.size(0)
            )
            
            net_xor.freeze_net()
            train_model(
                net_xor, 
                train_rxor_X, 
                train_rxor_y, 
                )
            predicted_label = net_xor(test_rxor_X)
            err_rxor_on_xor.append(
                1 - (torch.sigmoid(predicted_label).round() == test_rxor_y).sum().cpu().data.numpy().item() / test_rxor_y.size(0)
            )

            train_model(
                net_rxor, 
                train_rxor_X,
                train_rxor_y, 
                )
            predicted_label = net_rxor(test_rxor_X)
            err_rxor_on_rxor.append(
                1 - (torch.sigmoid(predicted_label).round() == test_rxor_y).sum().cpu().data.numpy().item() / test_rxor_y.size(0)
            )
            
            rep_list.append(ii+1)
            sample_list.append(sample)
            depth_list.append(depth)

summary['error xor on xor transformer'] = err_xor_on_xor
summary['error rxor on xor transformer'] = err_rxor_on_xor
summary['error rxor on rxor transformer'] = err_rxor_on_rxor
summary['rep'] = rep_list
summary['sample'] = sample_list
summary['depth'] = depth_list
summary.to_csv('./benchmarks/triplet_kalaeb.csv')
# %%
