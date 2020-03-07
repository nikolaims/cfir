import pandas as pd
import numpy as np
import h5py
from release.utils import band_hilbert, individual_max_snr_band
from release.constants import FS, N_SAMPLES_TEST, N_SAMPLES_TRAIN


# collect info
rest_df = pd.read_pickle('data/rest_10subjs_P4_500Hz_ICA_eye_rej.pkl')

eeg_df = pd.DataFrame(columns=['subj_id', 'snr', 'band_left_train', 'band_right_train', 'band_left', 'band_right', 'eeg',
                               'an_signal'])
eeg_df['an_signal'] = eeg_df['an_signal'].astype('complex')

# store data
for j_dataset, dataset in enumerate(rest_df['subj_id'].unique()):
    x = rest_df.loc[rest_df['subj_id']==j_dataset, 'P4']


    # drop noisy datasets
    if len(x) < N_SAMPLES_TRAIN+N_SAMPLES_TEST+10 * FS: continue



    # find individual alpha and snr
    band, snr = individual_max_snr_band(x, FS)
    print(j_dataset, band, snr)

    # save x
    an = band_hilbert(x, FS, band)[5 * FS:N_SAMPLES_TRAIN + N_SAMPLES_TEST+ 5 * FS]
    x = x[5 * FS:N_SAMPLES_TRAIN + N_SAMPLES_TEST+ 5 * FS]

    # train band
    band_train, snr_train = individual_max_snr_band(x[:N_SAMPLES_TRAIN], FS)
    print(j_dataset, band_train, snr_train, '\n')


    # save info
    band, snr = individual_max_snr_band(x[:N_SAMPLES_TRAIN], FS)
    eeg_df = eeg_df.append(pd.DataFrame({'subj_id': dataset, 'snr': snr, 'band_left_train': band_train[0],
                                         'band_right_train': band_train[1], 'band_left': band[0], 'band_right': band[1],
                                         'eeg': x, 'an_signal': an},),
                           ignore_index=True)


# save train test data
# eeg_df.to_pickle('data/rest_state_probes_real.pkl')


import pylab as plt
plt.plot(eeg_df['eeg'])