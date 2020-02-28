import pandas as pd
import numpy as np
import h5py
from release.utils import band_hilbert, individual_max_snr_band, load_data
from release.constants import FS, N_SAMPLES_TEST, N_SAMPLES_TRAIN, N_SUBJECTS
import pylab as plt
from release.filters import CFIRBandEnvelopeDetector
from scipy.linalg import eigh
import scipy.signal as sg

import sys
sys.path.insert(0, '/home/kolai/Projects/nfblab/nfb')
from pynfb.inlets.montage import Montage

from mne.viz import plot_topomap


# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('data/alpha_subject_2.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0] in ['FB0', 'FBMock', 'FB250', 'FB500'])][:]

eeg_df = pd.DataFrame(columns=['sim', 'dataset', 'snr', 'band_left_train', 'band_right_train', 'band_left',
                               'band_right', 'eeg', 'an_signal'])
eeg_df['an_signal'] = eeg_df['an_signal'].astype('complex')

# store data
dataset = datasets[7]
dataset_path = '{}/{}/experiment_data.h5'.format(data_path, dataset)

# load fb signal params
with h5py.File(dataset_path) as f:
    eye_rejection_matrix = f['protocol10/signals_stats/Alpha0/rejections/rejection1'].value

# load data
df, fs, channels, p_names = load_data(dataset_path)

# select baselines
df = df.loc[df['block_name'].isin(['Baseline0', 'Baseline'])].query('block_number<10')

# remove eyes artifacts ICA
df[channels] = df[channels].values.dot(eye_rejection_matrix)
channels = channels[:-1]
montage = Montage(channels)
x_multichannel = df[channels].values


n_fft = 500
n_ch = len(channels)
starts = np.arange(0, 10000, 500)


F_full = np.array([np.exp(-2j * np.pi / n_fft * k * np.arange(n_fft)) for k in np.arange(n_fft)])

C_k = np.zeros((n_fft, n_ch, n_ch), dtype=complex)
for k in range(n_fft):
    for m, start in enumerate(starts):
        slc = slice(start, start + n_fft)
        s = x_multichannel[slc]
        P_k = F_full[:, k].dot(s)
        C_k[k] += np.dot(P_k[:, None].conj(), P_k[None, :])
    C_k[k] /= len(starts)



delay = -20
cfir = CFIRBandEnvelopeDetector((8, 12), fs, delay, n_fft=n_fft)
b = cfir.b
freqs = np.arange(n_fft)
H = 2 * np.exp(-2j * np.pi * freqs / n_fft * delay)


bc = -1
for pp in range(3):
    fbmh = F_full.dot(b)-H
    C = np.sum([C_k[k]*np.abs(fbmh)[k] for k in range(n_fft)], 0).real
    lam, w = eigh(np.eye(n_ch), C+np.eye(n_ch)*0.01)

    xx = [w[bc].dot(C_k[k].real).dot(w[bc]) for k in range(n_fft)]
    cfir = CFIRBandEnvelopeDetector((8, 12), fs, delay, n_fft=n_fft, weights=xx)
    b = cfir.b
    bcorr = 0
    for c in range(32):
        y = band_hilbert(x_multichannel.dot(w[c]), fs, (8, 12))
        x = cfir.apply(x_multichannel.dot(w[c]))
        cor = np.corrcoef(np.roll(np.abs(y), delay), np.abs(x))[1, 0]
        if cor > bcorr:
            bcorr = cor
            bc = c

    cfir = CFIRBandEnvelopeDetector((8, 12), fs, delay, n_fft=n_fft, weights=xx)
    y = band_hilbert(x_multichannel[:, channels.index('P4')], fs, (8, 12))
    x = cfir.apply(x_multichannel[:, channels.index('P4')])
    print(bcorr, ';', np.corrcoef(np.roll(np.abs(y), delay), np.abs(x))[1, 0], lam[-1])

fig, axes = plt.subplots(4, 8)
for k in range(32):
    plot_topomap(x_multichannel.T.dot(x_multichannel).dot(w[k]), montage.get_pos(), axes=axes[k//8, k%8])

    y = band_hilbert(x_multichannel.dot(w[k]), fs, (8, 12))
    x = cfir.apply(x_multichannel.dot(w[k]))

    axes[k//8, k%8].set_xlabel('{:.2f}||{:.3f}'.format(lam[k], np.corrcoef(np.roll(np.abs(y), delay), np.abs(x))[1, 0]))
    if k == bc:
        axes[k // 8, k % 8].set_ylabel('BEST')


plt.figure()
plt.plot(x_multichannel.dot(w[bc]))
plt.plot(b)