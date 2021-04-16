#%%
import torch
import torch.nn.functional as F
from triplet import Net, train_model, predict
from triplet.utils import generate_gaussian_parity
import numpy as np
import pandas as pd
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
err_rxor_on_xor = []
err_rxor_on_rxor = []
rep_list = []
sample_list = []
depth_list = []
depths = [4,20]
widths = [3,20]

for depth,width in zip(depths,widths):
    for sample in sample_size:
        for ii in range(reps):
            print('doing sample %d rep %d'%(sample,ii+1))
            net_xor = Net(in_dim=2, out_dim=1, width=width, depth=depth, bias=True)
            net_rxor = Net(in_dim=2, out_dim=1, width=width, depth=depth, bias=True)
            
            #model_state_dic = torch.load('saved_model/model_weights_xor_depth_'+str(depth-1)+'_.pth',map_location=torch.device('cpu'))
            #net_xor.load_state_dict(model_state_dic)
            
            #source task
            train_xor_X, train_xor_y = generate_gaussian_parity(1000,cluster_std=0.25) 
            train_xor_X, train_xor_y = torch.FloatTensor(train_xor_X), (torch.FloatTensor(train_xor_y).unsqueeze(-1))
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
                train_xor_y,
                iteration=5000,
                verbose=False
                )
            train_model(
                net_xor, 
                train_rxor_X, 
                train_rxor_y, 
                iteration=5000,
                freeze=True, 
                verbose=False
                )

            predicted_label = predict(net_xor, test_rxor_X).view(test_size)
            err_rxor_on_xor.append(
                1 - np.mean(predicted_label.detach().numpy()==test_rxor_y)
            ) 

            train_model(
                net_rxor, 
                train_rxor_X,
                train_rxor_y, 
                iteration=5000,
                verbose=False
                )
            predicted_label = predict(net_rxor, test_rxor_X).view(test_size)
            err_rxor_on_rxor.append(
                1 - np.mean(predicted_label.detach().numpy()==test_rxor_y)
            )
            
            rep_list.append(ii+1)
            sample_list.append(sample)
            depth_list.append(depth)

summary['error rxor on xor transformer'] = err_rxor_on_xor
summary['error rxor on rxor transformer'] = err_rxor_on_rxor
summary['rep'] = rep_list
summary['sample'] = sample_list
summary['depth'] = depth_list
summary.to_csv('./benchmarks/triplet.csv')
# %%
