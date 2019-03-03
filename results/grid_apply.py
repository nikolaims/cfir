from pycfir.filters import RectEnvDetector, CFIRBandEnvelopeDetector, WHilbertFilter, FiltFiltARHilbertFilter, AdaptiveCFIRBandEnvelopeDetector
from pycfir.utils import individual_band_snr
import numpy as np
from settings import DELAY_RANGE, FS, N_SAMPLES_TRAIN, N_SAMPLES_TEST
import itertools
import pandas as pd
from tqdm import tqdm
from copy import deepcopy


def grid_apply_method(method_class, kwargs_grid):
    eeg_df = pd.read_pickle('data/rest_state_probes.pkl').query('dataset=="alpha2-delay-subj-1_11-06_17-15-29"')

    kwargs_grid['dataset'] = eeg_df['dataset'].unique()
    keys, values = zip(*kwargs_grid.items())
    kwargs_list = list(map(lambda x: dict(zip(keys, x)), itertools.product(*values)))

    res = np.empty((len(kwargs_list), N_SAMPLES_TRAIN+N_SAMPLES_TEST), dtype='complex64')
    for j_kwargs, kwargs in enumerate(tqdm(deepcopy(kwargs_list), mininterval=1, desc=method_class.__name__)):
        df = eeg_df.query('dataset=="{}"'.format(kwargs.pop('dataset')))
        x = df['eeg'].values
        band = (df['band_left_train'].values[0], df['band_right_train'].values[0])
        try:
            method = method_class(band=band, fs=FS, **kwargs)
            y = method.apply(x[:N_SAMPLES_TRAIN])
            res[j_kwargs, :N_SAMPLES_TRAIN] = y

            method = method_class(band=band, fs=FS, **kwargs)
            y = method.apply(x[N_SAMPLES_TRAIN:N_SAMPLES_TRAIN+N_SAMPLES_TEST])
            res[j_kwargs, N_SAMPLES_TRAIN:N_SAMPLES_TRAIN+N_SAMPLES_TEST] = y
        except ValueError:
            res[j_kwargs, :] = np.nan

    return res, pd.DataFrame(kwargs_list)



# Rect env detector
kwargs_grid = {
    'n_taps_bandpass': np.arange(0, DELAY_RANGE.max()+1, np.diff(DELAY_RANGE)[0]//2)*2,
    'delay': np.arange(0, DELAY_RANGE.max()+1, np.diff(DELAY_RANGE)[0])
}
res, kwargs_df = grid_apply_method(RectEnvDetector, kwargs_grid)
np.save('results/rect.npy', res)
kwargs_df.to_csv('results/rect_kwargs.csv', index=False)


# cFIR
kwargs_grid = {
    'delay': DELAY_RANGE,
    'n_taps': [300, 500, 1000]
}
res, kwargs_df = grid_apply_method(CFIRBandEnvelopeDetector, kwargs_grid)
np.save('results/cfir.npy', res)
kwargs_df.to_csv('results/cfir_kwargs.csv', index=False)


#wHilbert
kwargs_grid = {
    'delay': np.arange(DELAY_RANGE[DELAY_RANGE>0].min(), DELAY_RANGE.max()+1, np.diff(DELAY_RANGE)[0]),
    'n_taps': np.arange(200, 2000 + 1, 100),
    'max_chunk_size': [10]
}

res, kwargs_df = grid_apply_method(WHilbertFilter, kwargs_grid)
np.save('results/whilbert.npy', res)
kwargs_df.to_csv('results/whilbert_kwargs.csv',index=False)


#ffiltAR
kwargs_grid = {
    'delay': DELAY_RANGE,
    'n_taps_edge': np.arange(10, 50, 10, dtype=int),
    'ar_order': [50, 100],
    'max_chunk_size': [10],
    'buffer_s': [4]
}

res, kwargs_df = grid_apply_method(FiltFiltARHilbertFilter, kwargs_grid)
np.save('results/ffiltar.npy', res)
kwargs_df.to_csv('results/ffiltar_kwargs.csv',index=False)

# rls

kwargs_grid = {
    'delay': DELAY_RANGE,
    'n_taps': [300],#np.arange(200, 2000 + 1, 300),
    'ada_n_taps': [300],
    'mu': [0.9],
    'max_chunk_size': [100]
}

res, kwargs_df = grid_apply_method(AdaptiveCFIRBandEnvelopeDetector, kwargs_grid)
np.save('results/rlscfir.npy', res)
kwargs_df.to_csv('results/rlscfir_kwargs.csv',index=False)