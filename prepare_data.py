"""
Prepare train test data
"""

import pandas as pd
import numpy as np
from utils import band_hilbert, individual_max_snr_band
from constants import FS, N_SAMPLES_TEST, N_SAMPLES_TRAIN


# collect info
rest_df = pd.read_pickle('data/rest_10subjs_32ch_500Hz_ICA_eye_rej.pkl')
channels = list(rest_df.columns[:-1])

eeg_df = pd.DataFrame(columns=channels + ['subj_id', 'snr', 'band_left_train', 'band_right_train',
                                                 'band_left', 'band_right', 'eeg', 'an_signal'])

eeg_df['an_signal'] = eeg_df['an_signal'].astype('complex')

# store data
for j_dataset, dataset in enumerate(rest_df['subj_id'].unique()):

    x = rest_df.loc[rest_df['subj_id'] == j_dataset, channels]
    p4 = x['P4'].values

    # find individual alpha and snr
    band, snr = individual_max_snr_band(p4, FS)
    print(j_dataset, band, snr)

    # save x
    slc = slice(5 * FS, N_SAMPLES_TRAIN + N_SAMPLES_TEST+ 5 * FS)
    an = band_hilbert(p4, FS, band)[slc]
    x = x.values[slc]
    p4 = p4[slc]

    # swap train test
    # an = np.concatenate([an[N_SAMPLES_TRAIN:N_SAMPLES_TRAIN + N_SAMPLES_TEST], an[:N_SAMPLES_TRAIN]])
    # x = np.concatenate([x[N_SAMPLES_TRAIN:N_SAMPLES_TRAIN + N_SAMPLES_TEST], x[:N_SAMPLES_TRAIN]])

    # train band
    band_train, snr_train = individual_max_snr_band(p4[:N_SAMPLES_TRAIN], FS)
    print(j_dataset, band_train, snr_train, '\n')


    # save info
    band, snr = individual_max_snr_band(p4, FS)
    eeg_df_subj = pd.DataFrame(data=x, columns=rest_df.columns[:-1])
    eeg_df_subj['subj_id'] = dataset
    eeg_df_subj['snr'] = snr
    eeg_df_subj['band_left_train'] = band_train[0]
    eeg_df_subj['band_right_train'] = band_train[1]
    eeg_df_subj['band_left'] = band[0]
    eeg_df_subj['band_right'] = band[1]
    eeg_df_subj['eeg'] = 'P4'
    eeg_df_subj['an_signal'] = an
    eeg_df = eeg_df.append(eeg_df_subj, ignore_index=True)


# save train test data
eeg_df.to_pickle('data/train_test_data_multichannel.pkl')


import pylab as plt
plt.plot(eeg_df['P4'])