import pandas as pd
import numpy as np
import scipy.signal as sg
import sys
import h5py

# import nfb lab data loader
sys.path.insert(0, '/home/kolai/Projects/nfblab/nfb')
from utils.load_results import load_data


FLANKER_WIDTH = 2
FS = 500
GFP_THRESHOLD = 100e-6

# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('data/alpha_subject_2.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0] in ['FB0', 'FBMock', 'FB250', 'FB500'])][:]

# store data
probes = []
for j_dataset, dataset in enumerate(datasets):
    dataset_path = '{}/{}/experiment_data.h5'.format(data_path, dataset)

    # load fb signal params
    with h5py.File(dataset_path) as f:
        eye_rejection_matrix = f['protocol10/signals_stats/Alpha0/rejections/rejection1'].value
        band = f['protocol10/signals_stats/Alpha0/bandpass'].value

    # load data
    df, fs, channels, p_names = load_data(dataset_path)

    # select baselines
    df = df.loc[df['block_name'].isin(['Baseline0', 'Baseline'])].query('block_number<10')

    # remove eyes artifacts ICA
    df[channels] = df[channels].values.dot(eye_rejection_matrix)

    # GFP threshold artifact segments
    th = np.abs(df[channels[:-1]]).rolling(int(fs), center=True).max().mean(1)
    df = df.loc[th<GFP_THRESHOLD]

    # define SNR
    x = df['P4'].values[10*FS:]
    freq, pxx = sg.welch(x, FS, nperseg=FS * 2)
    sig = pxx[(freq >= band[0]) & (freq <= band[1])].mean()
    noise = pxx[((freq >= band[0] - FLANKER_WIDTH) & (freq <= band[0])) | (
            (freq >= band[1]) & (freq <= band[1] + FLANKER_WIDTH))].mean()
    snr = sig / noise

    # drop low SNR
    if snr < 1: continue
    # drop noisy datasets
    if len(x) < 160*FS: continue

    # save x
    x = x[:160*FS]
    probes.append(x)

# save probes
probes = np.array(probes)
np.save('data/rest_state_probes.npy', probes)