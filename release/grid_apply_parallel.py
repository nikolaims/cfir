from release.filters import RectEnvDetector, CFIRBandEnvelopeDetector, AdaptiveCFIRBandEnvelopeDetector
import numpy as np
from release.constants import DELAY_RANGE, FS, N_SAMPLES_TRAIN, N_SAMPLES_TEST
import itertools
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import time
import multiprocessing


kwargs_grid_dict = {}

# rls
kwargs_grid_dict['rlscfir'] = (AdaptiveCFIRBandEnvelopeDetector, {
    'delay': DELAY_RANGE,
    'n_taps': [500],#np.arange(200, 2000 + 1, 300),
    'ada_n_taps': [5000],
    'mu': [0.9, 0.8],
})


# cFIR
kwargs_grid_dict['cfir'] = (CFIRBandEnvelopeDetector, {
    'delay': DELAY_RANGE,
    'n_taps': [500]
})


# rect
kwargs_grid_dict['rect'] = (RectEnvDetector, {
    'n_taps_bandpass': np.arange(0, DELAY_RANGE.max()+1, np.diff(DELAY_RANGE)[0]//2)*2,
    'delay': np.arange(0, DELAY_RANGE.max()+1, np.diff(DELAY_RANGE)[0])
})


eeg_df = pd.read_pickle('data/rest_state_probes_real.pkl')#.query('dataset=="alpha2-delay-subj-1_11-06_17-15-29"')


for method_name in kwargs_grid_dict:

    start = time.time()
    method_class, kwargs_grid = kwargs_grid_dict[method_name]

    kwargs_grid['dataset'] = eeg_df['dataset'].unique()
    keys, values = zip(*kwargs_grid.items())
    kwargs_list = list(map(lambda x: dict(zip(keys, x)), itertools.product(*values)))
    pbar = tqdm(total=len(kwargs_list)//4, desc=method_name)

    def grid_apply_method(method_kwargs):
        global method_class
        res = np.empty(N_SAMPLES_TRAIN + N_SAMPLES_TEST, dtype='complex64')
        method_kwargs = deepcopy(method_kwargs)
        df = eeg_df.query('dataset=="{}"'.format(method_kwargs['dataset']))
        x = df['eeg'].values
        band = (df['band_left_train'].values[0], df['band_right_train'].values[0])
        try:
            method = method_class(band=band, fs=FS, **method_kwargs)
            y = method.apply(x[:N_SAMPLES_TRAIN])
            res[:N_SAMPLES_TRAIN] = y

            method = method_class(band=band, fs=FS, **method_kwargs)
            y = method.apply(x[N_SAMPLES_TRAIN:N_SAMPLES_TRAIN+N_SAMPLES_TEST])
            res[N_SAMPLES_TRAIN:N_SAMPLES_TRAIN+N_SAMPLES_TEST] = y
        except ValueError:
            res[:] = np.nan
        pbar.update(1)
        return res, method_kwargs


    pool = multiprocessing.Pool(4)
    res, kwargs_df = zip(*pool.map(grid_apply_method, kwargs_list))


    np.save('results/{}.npy'.format(method_name), res)
    pd.DataFrame(list(kwargs_df)).to_csv('results/{}_kwargs.csv'.format(method_name), index=False)

    print(method_name, time.time() - start)