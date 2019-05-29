import pandas as pd
import numpy as np
import sys
import h5py
from release.utils import band_hilbert, individual_max_snr_band
from release.constants import FS, N_SAMPLES_TEST, N_SAMPLES_TRAIN, N_SUBJECTS

# import nfb lab data loader
sys.path.insert(0, '/home/kolai/Projects/nfblab/nfb')
from utils.load_results import load_data

# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('data/alpha_subject_2.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0] in ['FB0', 'FBMock', 'FB250', 'FB500'])][:]

eeg_df = pd.DataFrame(columns=['sim', 'dataset', 'snr', 'band_left_train', 'band_right_train', 'eeg', 'an_signal'])
eeg_df['an_signal'] = eeg_df['an_signal'].astype('complex')

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
    # th = np.abs(df[channels[:-1]]).rolling(int(fs), center=True).mean().mean(1)
    # df = df.loc[th < 100e-6]

    # define SNR
    x = df['P4'].values[5 * FS:]

    # find individual alpha and snr
    band, snr = individual_max_snr_band(x, FS)
    print(band, snr)

    # drop noisy datasets
    if len(x) < N_SAMPLES_TRAIN+N_SAMPLES_TEST+10 * FS: continue

    # save x
    an = band_hilbert(x, fs, band)[5 * FS:N_SAMPLES_TRAIN + N_SAMPLES_TEST+ 5 * FS]
    x = x[5 * FS:N_SAMPLES_TRAIN + N_SAMPLES_TEST+ 5 * FS]

    # save info
    band, snr = individual_max_snr_band(x[:N_SAMPLES_TRAIN], FS)
    eeg_df = eeg_df.append(pd.DataFrame({'sim': 0, 'dataset': dataset, 'snr': snr,
                                 'band_left_train': band[0], 'band_right_train': band[1], 'eeg': x, 'an_signal': an},),
                           ignore_index=True)




# select subjects
from sklearn.cluster import KMeans

datasets = eeg_df['dataset'].unique()
snrs = np.array([eeg_df.query('dataset=="{}"'.format(d)).snr.values[0] for d in datasets])

k_means = KMeans(n_clusters=N_SUBJECTS, random_state=42)
k_means.fit(snrs[:, None])

subjects = [datasets[k_means.labels_==k][0] for k in range(N_SUBJECTS)]


eeg_df = eeg_df.loc[eeg_df['dataset'].isin(subjects)]
eeg_df = eeg_df.reset_index(drop=True)

# save info
eeg_df.to_pickle('data/rest_state_probes_real.pkl')


import pylab as plt
plt.plot(eeg_df['eeg'])