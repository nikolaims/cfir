import itertools
import multiprocessing
import numpy as np
import pandas as pd

from tqdm import tqdm
from copy import deepcopy

from release.filters import \
    RectEnvDetector, CFIRBandEnvelopeDetector, AdaptiveCFIRBandEnvelopeDetector, FiltFiltARHilbertFilter
from release.constants import DELAY_RANGE, FS, N_SAMPLES_TRAIN, N_SAMPLES_TEST
from release.utils import magnitude_spectrum


# load data
eeg_df = pd.read_pickle('data/train_test_data.pkl')

# compute train magnitude spectrum for weights if it needed
magnitude_spectrum_train = {}
for dataset, dataset_df in eeg_df.groupby('subj_id'):
    _, weights = magnitude_spectrum(eeg_df['eeg'].values[:N_SAMPLES_TRAIN], FS)
    magnitude_spectrum_train[dataset] = weights


# init grid search space
kwargs_grid_dict = {}

# rls
kwargs_grid_dict['rlscfir'] = (AdaptiveCFIRBandEnvelopeDetector, {
    'delay': DELAY_RANGE,
    'n_taps': [250, 500, 1000],#np.arange(200, 2000 + 1, 300),
    'ada_n_taps': [5000],
    'mu': [0.7, 0.8, 0.9],
    'upd_samples': [25, 50],
})


# cFIR
kwargs_grid_dict['cfir'] = (CFIRBandEnvelopeDetector, {
    'delay': DELAY_RANGE,
    'n_taps': [250, 500, 1000],
    'weights': [None]
})

# wcFIR
kwargs_grid_dict['wcfir'] = (CFIRBandEnvelopeDetector, {
    'delay': DELAY_RANGE,
    'n_taps': [250, 500, 1000],
    'weights': [True]
})

# rect
kwargs_grid_dict['rect'] = (RectEnvDetector, {
    'n_taps_bandpass': np.arange(0, DELAY_RANGE.max()+1, 5)*2,
    'delay': np.arange(0, DELAY_RANGE.max()+1, np.diff(DELAY_RANGE)[0])
})


# ffiltar
kwargs_grid_dict['ffiltar'] = (FiltFiltARHilbertFilter, {
    'delay': [0],
    'n_taps': [2000],
    'ar_order': [10, 25, 50],
    'n_taps_edge': [10, 25,  50],
    'butter_order': [1, 2]
})

# loop over methods
for method_name in kwargs_grid_dict:
    # unpack class and grid params
    method_class, kwargs_grid = kwargs_grid_dict[method_name]

    # add datasets to params
    kwargs_grid['subj_id'] = eeg_df['subj_id'].unique()

    # all combinations of params to list
    keys, values = zip(*kwargs_grid.items())
    kwargs_list = list(map(lambda x: dict(zip(keys, x)), itertools.product(*values)))

    # tqdm status monitor settings
    pbar = tqdm(total=len(kwargs_list)//4, desc=method_name)

    # method to parallel computing
    def grid_apply_method(method_kwargs):
        # init results array
        res = np.empty(N_SAMPLES_TRAIN + N_SAMPLES_TEST, dtype='complex64')
        # method params
        method_kwargs = deepcopy(method_kwargs)

        # get x and train band
        df = eeg_df.query('subj_id=={}'.format(method_kwargs['subj_id']))
        x = df['eeg'].values
        band = (df['band_left_train'].values[0], df['band_right_train'].values[0])

        # handle weights logic
        if 'weights' in method_kwargs:
            if method_kwargs['weights'] is not None:
                method_kwargs['weights'] = magnitude_spectrum_train[method_kwargs['subj_id']]

        # train data
        method = method_class(band=band, fs=FS, **method_kwargs)
        x_slice = x[:N_SAMPLES_TRAIN]
        res[:N_SAMPLES_TRAIN] = method.apply(x_slice)

        # test data
        method = method_class(band=band, fs=FS, **method_kwargs)
        x_slice = x[N_SAMPLES_TRAIN:N_SAMPLES_TRAIN+N_SAMPLES_TEST]
        res[N_SAMPLES_TRAIN:N_SAMPLES_TRAIN+N_SAMPLES_TEST] = method.apply(x_slice)

        # update tqdm bar
        pbar.update(1)
        return res, method_kwargs

    # run grid search
    pool = multiprocessing.Pool(4)
    res, kwargs_df = zip(*pool.map(grid_apply_method, kwargs_list))

    # save estimations
    np.save('results/{}.npy'.format(method_name), res)

    # save list of all params combinations
    pd.DataFrame(list(kwargs_df)).to_csv('results/{}_kwargs.csv'.format(method_name), index=False)
