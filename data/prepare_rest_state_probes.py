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
ALPHA_BAND_EXT = (7, 13)
ALPHA_BAND_HALFWIDTH = 2

# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('data/alpha_subject_2.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0] in ['FB0', 'FBMock', 'FB250', 'FB500'])][:]

eeg_df = pd.DataFrame(columns=['sim', 'dataset', 'snr', 'band_left', 'band_right', 'eeg'])

# store data
for j_dataset, dataset in enumerate(datasets):
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

    # GFP threshold artifact segments
    th = np.abs(df[channels[:-1]]).rolling(int(fs), center=True).max().mean(1)
    df = df.loc[th<GFP_THRESHOLD]

    # define SNR
    x = df['P4'].values[10*FS:]
    freq, pxx = sg.welch(x, FS, nperseg=FS * 2)

    # find individual alpha
    alpha_mask = np.logical_and(freq > ALPHA_BAND_EXT[0], freq < ALPHA_BAND_EXT[1])
    main_freq = freq[alpha_mask][np.argmax(pxx[alpha_mask])]
    band = (main_freq-ALPHA_BAND_HALFWIDTH, main_freq+ALPHA_BAND_HALFWIDTH)
    sig = pxx[(freq >= band[0]) & (freq <= band[1])].mean()
    noise = pxx[((freq >= band[0] - FLANKER_WIDTH) & (freq <= band[0])) | (
            (freq >= band[1]) & (freq <= band[1] + FLANKER_WIDTH))].mean()
    snr = sig / noise

    # drop noisy datasets
    if len(x) < 160*FS: continue

    # save info
    eeg_df = eeg_df.append(pd.DataFrame({'sim': 0, 'dataset': dataset, 'snr': snr,
                                 'band_left': band[0], 'band_right': band[1], 'eeg': x}), ignore_index=True)

    # save x
    x = x[:160*FS]




# select subjects
N_SUBJECTS = 10
bins = np.linspace(eeg_df['snr'].min(), eeg_df['snr'].max(), N_SUBJECTS + 1)
subjects = []
for snr_left, snr_right in zip(bins[:-1], bins[1:]):
    subjects.append(eeg_df.query('snr>{} & snr<={}'.format(snr_left, snr_right))['dataset'].values[0])

eeg_df = eeg_df.loc[eeg_df['dataset'].isin(subjects)]

# save info
eeg_df.to_pickle('data/rest_state_probes_info.pkl')