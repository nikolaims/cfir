import pandas as pd
import numpy as np
import h5py
from release.utils import band_hilbert, individual_max_snr_band, load_data
from release.constants import FS, N_SAMPLES_TEST, N_SAMPLES_TRAIN, N_SUBJECTS


groups_to_analize = ['FB0', 'FBMock']

# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('experiments/prestates/alpha_subject_2_full.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0] in groups_to_analize)][:]

print(len(datasets))

eeg_df = pd.DataFrame(columns=['subj_id', 'P4', 'block_number'])
info_df = pd.DataFrame(columns=['subj_id', 'dataset', 'band_l', 'band_r', 'fb_type'])

# store data
for j_dataset, dataset in enumerate(datasets):
    dataset_path = '{}/{}/experiment_data.h5'.format(data_path, dataset)

    # load fb signal params
    with h5py.File(dataset_path) as f:
        eye_rejection_matrix = f['protocol10/signals_stats/Alpha0/rejections/rejection1'].value
        band = f['protocol10/signals_stats/Alpha0/bandpass'].value

    # load data
    df, fs, channels, p_names = load_data(dataset_path)

    # select baselines
    df = df.loc[df['block_name'].isin(groups_to_analize)]
    fb_type = df['block_name'].values[0]

    # remove eyes artifacts ICA
    df[channels] = df[channels].values.dot(eye_rejection_matrix)

    # GFP threshold artifact segments
    th = np.abs(df['P4']).rolling(int(fs), center=True).max()
    df = df.loc[th < 100e-6]

    # define SNR
    x = df['P4'].values
    block_numbers_dict = dict(zip(df['block_number'].unique(), np.arange(len(df['block_number'].unique()))))

    block_number = df['block_number'].map(block_numbers_dict)

    eeg_df = eeg_df.append(pd.DataFrame({'subj_id': j_dataset, 'P4': x, 'block_number': block_number}),
                           ignore_index=True)
    info_df = info_df.append({'subj_id': j_dataset, 'dataset': dataset, 'band_l': band[0], 'band_r': band[1], 'fb_type': fb_type},
                           ignore_index=True)


    print(eeg_df.tail())
    print(info_df.tail())

eeg_df.to_pickle('experiments/prestates/data.pkl')
info_df.to_pickle('experiments/prestates/info.csv')