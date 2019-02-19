import pandas as pd
import numpy as np
import sys
import h5py
import pylab as plt
import scipy.signal as sg
from mne.viz import plot_topomap

# import nfb lab data loader
sys.path.insert(0, '/home/kolai/Projects/nfblab/nfb')
from utils.load_results import load_data
from pynfb.inlets.montage import Montage
from pynfb.signal_processing.decompositions import ICADecomposition

df, fs, channels, p_names = load_data('/media/kolai/OS/Projects/nfblab/nfb/pynfb/results/vr1_02-18_15-30-10/experiment_data.h5')
df = df.query('block_number<=2')
df2, fs_, channels_, p_names_ = load_data('/media/kolai/OS/Projects/nfblab/nfb/pynfb/results/vr1_02-18_15-52-17/experiment_data.h5')
df2['block_number'] += 4
channels = channels[:30]
montage = Montage(channels)

df = df.append(df2, ignore_index=True)

b, a = sg.butter(4, 3/(fs/2), 'highpass')
df[channels] = sg.filtfilt(b, a, df[channels], axis=0)

# plt.plot(df['C3'])
# plt.show()
#
# freq, pxx = sg.welch(df['C3'], fs)
# plt.plot(freq, pxx)


alpha_pows = []
for j, block_name in enumerate(df['block_name'].unique()):
    freq, pxx = sg.welch(df.query('block_name=="{}"'.format(block_name))[channels], fs, nperseg=fs, axis=0)
    alpha_pow = pxx[(freq > 8) & (freq < 14)].mean(0)
    alpha_pows.append(alpha_pow)
    #plt.plot(freq, pxx, label=block_name)
    #plt.plot(df.query('block_name=="{}"'.format(block_name))['C3'])
    # plot_topomap(alpha_pow, montage.get_pos(), axes=axes[j], show=False)
    # axes[j].set_title(block_name)


fig, axes = plt.subplots(1, 3)
ers_monitor = (alpha_pows[1]-alpha_pows[0])/alpha_pows[0]
ers_vr = (alpha_pows[3]-alpha_pows[2])/alpha_pows[2]
plot_topomap(ers_monitor, montage.get_pos(), axes=axes[0], show=False, vmin=-0.5, vmax=0.5)
plot_topomap(ers_vr, montage.get_pos(), axes=axes[1], show=False, vmin=-0.5, vmax=0.5)
plot_topomap(ers_vr- ers_monitor, montage.get_pos(), axes=axes[2], show=False, vmin=-0.5, vmax=0.5)
#plot_topomap(, montage.get_pos(), axes=axes[1], show=False, vmin=-1, vmax=1)

plt.legend()



ica = ICADecomposition(channels, fs)
ica.fit(df[channels].values)

fig, axes = plt.subplots(1, 30)
for j, topo in enumerate(ica.topographies.T):
    plot_topomap(topo, montage.get_pos(), axes=axes[j], show=False)
    axes[j].set_title(j)

df_sources = pd.DataFrame(columns=['source{}'.format(k) for k in range(30)], data=df[channels].values.dot(ica.filters))
df_sources['block_name'] = df['block_name'].values



alpha_pows = []
fig, axes = plt.subplots(15, 4)#, sharex=True)
plt.subplots_adjust(hspace=0)
for k in range(30):
    for j, block_name in enumerate(df_sources['block_name'].unique()):
        freq, pxx = sg.welch(df_sources.query('block_name=="{}"'.format(block_name))['source{}'.format(k)], fs, nperseg=fs, axis=0)
        axes[k%15, k//15*2+1].set_xlim(0, 60)
        axes[k % 15, k // 15 * 2 + 1].semilogy(freq, pxx, label=block_name)

        plot_topomap(ica.topographies[:, k], montage.get_pos(), axes=axes[k%15, k//15*2], show=False)

    #plt.plot(df.query('block_name=="{}"'.format(block_name))['C3'])
    # plot_topomap(alpha_pow, montage.get_pos(), axes=axes[j], show=False)
    # axes[j].set_title(block_name)

axes[-1, 1].set_xlabel('Freq. Hz')
axes[-1, 3].set_xlabel('Freq. Hz')
axes[0, 1].legend()