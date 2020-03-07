import pandas as pd
import numpy as np
import h5py
from release.utils import band_hilbert, individual_max_snr_band, load_data
from release.constants import FS, N_SAMPLES_TEST, N_SAMPLES_TRAIN, N_SUBJECTS

subjects = ['alpha2-delay-subj-1_11-06_17-15-29', 'alpha2-delay-subj-2_11-07_17-06-03',
            'alpha2-delay-subj-4_11-12_11-58-16', 'alpha2-delay-subj-5_11-12_20-56-08',
            'alpha2-delay-subj-6_11-14_11-06-10', 'alpha2-delay-subj-7_11-15_11-38-15',
            'alpha2-delay-subj-8_11-15_17-55-21', 'alpha2-delay-subj-11_11-21_18-56-34',
            'alpha2-delay-subj-21_12-06_12-15-09', 'alpha2-delay-subj-28_12-14_17-19-21']



CHANNELS = ['FP1', 'FP2', 'F7', 'F3', 'FZ', 'F4', 'F8', 'FT9', 'FC5', 'FC1', 'FC2', 'FC6', 'FT10', 'C3', 'CZ', 'C4',
            'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'P4', 'P8', 'O1', 'OZ', 'O2', 'T7', 'PZ']

# collect info
data_path = '/home/kolai/Data/alpha_delay2'
info = pd.read_csv('data/alpha_subject_2.csv')
datasets = [d for d in info['dataset'].unique() if (d is not np.nan)
            and (info.query('dataset=="{}"'.format(d))['type'].values[0] in ['FB0', 'FBMock', 'FB250', 'FB500'])][:]

eeg_df = pd.DataFrame(columns=CHANNELS+['subj_id'])
# store data
for j_dataset, dataset in enumerate(subjects):
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
    x = df[CHANNELS].values[5 * FS:]
    eeg_df_subj = pd.DataFrame(data = x, columns=CHANNELS)
    eeg_df_subj['subj_id'] = j_dataset
    eeg_df = eeg_df.append(eeg_df_subj, ignore_index=True)

eeg_df.to_pickle('data/rest_10subjs_32ch_500Hz_ICA_eye_rej.pkl')
eeg_df[['subj_id', 'P4']].to_pickle('data/rest_10subjs_P4_500Hz_ICA_eye_rej.pkl')