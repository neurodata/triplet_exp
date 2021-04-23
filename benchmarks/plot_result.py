#%%
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
#%%
df = pd.read_csv('triplet_kalaeb.csv')
reps = 20
sample_size = np.logspace(
        np.log10(1e2),
        np.log10(1e4),
        num=10,
        endpoint=True,
        dtype=int
        )

err_rxor1 = np.zeros((reps,len(sample_size)), dtype=float)
err_rxor2 = np.zeros((reps,len(sample_size)), dtype=float)
err_rxor1_ = np.zeros((reps,len(sample_size)), dtype=float)
err_rxor2_ = np.zeros((reps,len(sample_size)), dtype=float)

df_ = df[df['depth']==3]
for ii,sample in enumerate(sample_size):
    err_rxor1[:,ii] = df_['error rxor on xor transformer'][df['sample']==sample]
    err_rxor2[:,ii] = df_['error rxor on rxor transformer'][df['sample']==sample]

df__ = df[df['depth']==19]
for ii,sample in enumerate(sample_size):
    err_rxor1_[:,ii] = df__['error rxor on xor transformer'][df['sample']==sample]
    err_rxor2_[:,ii] = df__['error rxor on rxor transformer'][df['sample']==sample]
# %%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))

ax.plot(sample_size, np.mean(err_rxor1,axis=0), c='k', label='error rxor on xor transformer shallow net')
ax.plot(sample_size, np.mean(err_rxor2,axis=0), c='r', label='error rxor on rxor transformer shallow net')
ax.plot(sample_size, np.mean(err_rxor1_,axis=0), marker='.', c='k', label='error rxor on xor transformer deeper net')
ax.plot(sample_size, np.mean(err_rxor2_,axis=0), marker='.', c='r', label='error rxor on rxor transformer deeper net')
    #ax.fill_between(sample_size, mean_kdf-1.96*var_kdf, mean_kdf+1.96*var_kdf, facecolor='r', alpha=0.5)
    #ax[0].plot(sample_size, np.mean(kappa_rf,axis=1), label='RF', c='k', lw=3)
    #ax.fill_between(sample_size, mean_rf-1.96*var_kdf, mean_rf+1.96*var_kdf, facecolor='k', alpha=0.5)

ax.set_xlabel('Sample size')
ax.set_ylabel('error')
ax.set_xscale('log')
ax.legend(frameon=False)
#ax[0][0].set_yticks([0,.2,.4,.6,.8,1])
right_side = ax.spines["right"]
right_side.set_visible(False)
top_side = ax.spines["top"]
top_side.set_visible(False)

plt.savefig('plots/triplet.pdf')
# %%
