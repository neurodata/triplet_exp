#%%
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
from triplet.dd import *
#%%
df = pd.read_csv('triplet_kalaeb_100_5.csv')
reps = 100
sample_size = np.logspace(
        np.log10(1e1),
        np.log10(1e3),
        num=5,
        endpoint=True,
        dtype=int
        )

err_rxor1 = np.zeros((reps,len(sample_size)), dtype=float)
err_rxor2 = np.zeros((reps,len(sample_size)), dtype=float)

for ii,sample in enumerate(sample_size):
    err_rxor1[:,ii] = df['error rxor on xor transformer 1'][df['sample']==sample]
    err_rxor2[:,ii] = df['error rxor on xor transformer 2'][df['sample']==sample]
# %%
sns.set_context('talk')
fig, ax = plt.subplots(1,1, figsize=(8,8))
qunatiles = np.nanquantile(err_rxor1,[.25,.75],axis=0)
ax.fill_between(sample_size, qunatiles[0], qunatiles[1], facecolor='k', alpha=.3)
ax.plot(sample_size, np.mean(err_rxor1,axis=0), c='k', label='shallow net (49 polytopes)')

qunatiles = np.nanquantile(err_rxor2,[.25,.75],axis=0)
ax.fill_between(sample_size, qunatiles[0], qunatiles[1], facecolor='r', alpha=.3)
ax.plot(sample_size, np.mean(err_rxor2,axis=0), c='r', label='deep net (363 polytopes)')
#ax.plot(sample_size, np.mean(err_rxor1_,axis=0), marker='.', c='k', label='error rxor on xor transformer deeper net')
#ax.plot(sample_size, np.mean(err_rxor2_,axis=0), marker='.', c='r', label='error rxor on rxor transformer deeper net')
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
def get_colors(colors, inds):
    c = [colors[i] for i in inds]
    return c

fontsize = 30
labelsize = 28


fig,ax = plt.subplots(1,1, figsize=(8,8))
colors = sns.color_palette("Dark2", n_colors=2)

X, Y, _, _ = get_dataset(N=1000, cov_scale=.5)
Y = Y.view(-1).numpy().astype(int)

ax.scatter(X[:, 0], X[:, 1], c=get_colors(colors, Y), s=50)

ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Gaussian XOR", fontsize=30)

plt.tight_layout()
ax.axis("off")
# %%
