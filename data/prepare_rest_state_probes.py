import pandas as pd
import numpy as np
import sys
import h5py
from data.settings import FLANKER_WIDTH, FS, GFP_THRESHOLD, ALPHA_BAND_EXT, ALPHA_BAND_HALFWIDTH, N_SUBJECTS
from pycfir.filters import band_hilbert
from pycfir.utils import individual_band_snr

# import nfb lab data loader
sys.path.insert(0, '/home/kolai/Projects/nfblab/nfb')
from utils.load_results import load_data

# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('data/alpha_subject_2.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0] in ['FB0', 'FBMock', 'FB250', 'FB500'])][:]

eeg_df = pd.DataFrame(columns=['sim', 'dataset', 'snr', 'band_left', 'band_right', 'eeg', 'an_signal'])
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
    th = np.abs(df[channels[:-1]]).rolling(int(fs), center=True).max().mean(1)
    df = df.loc[th < GFP_THRESHOLD]

    # define SNR
    x = df['P4'].values[5 * FS:]

    # find individual alpha and snr
    band, snr = individual_band_snr(x, FS, ALPHA_BAND_EXT, ALPHA_BAND_HALFWIDTH, FLANKER_WIDTH)
    print(band, snr)

    # drop noisy datasets
    if len(x) < 170* FS: continue

    # save x
    an = band_hilbert(x, fs, band)[5 * FS:165 * FS]
    x = x[5 * FS:165 * FS]

    # save info
    eeg_df = eeg_df.append(pd.DataFrame({'sim': 0, 'dataset': dataset, 'snr': snr,
                                 'band_left': band[0], 'band_right': band[1], 'eeg': x, 'an_signal': an},),
                           ignore_index=True)




# select subjects
gran = 2
bins = np.linspace(eeg_df['snr'].min(), eeg_df['snr'].max(), N_SUBJECTS//gran + 1)
subjects = []
for snr_left, snr_right in zip(bins[:-1], bins[1:]):
    for g in range(gran):
        subjects.append(eeg_df.query('snr>{} & snr<={}'.format(snr_left, snr_right))['dataset'].unique()[g])

eeg_df = eeg_df.loc[eeg_df['dataset'].isin(subjects)]

# save info
eeg_df.to_pickle('data/rest_state_probes_info.pkl')