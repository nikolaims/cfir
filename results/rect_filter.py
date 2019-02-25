from pycfir.filters import RectEnvDetector, CFIRBandEnvelopeDetector
from pycfir.utils import individual_band_snr
import numpy as np
from settings import DELAY_RANGE, FS, N_SAMPLES_TRAIN, N_SAMPLES_TEST
import itertools
import pandas as pd
from tqdm import tqdm


def grid_apply_method(method_class, kwargs_grid):
    eeg_df = pd.read_pickle('data/rest_state_probes_real.pkl')

    kwargs_grid['dataset'] = eeg_df['dataset'].unique()
    keys, values = zip(*kwargs_grid.items())
    kwargs_list = list(map(lambda x: dict(zip(keys, x)), itertools.product(*values)))

    res = np.empty((len(kwargs_list), N_SAMPLES_TRAIN+N_SAMPLES_TEST))
    for j_kwargs, kwargs in enumerate(tqdm(kwargs_list, mininterval=1)):
        x = eeg_df.query('dataset=="{}"'.format(kwargs.pop('dataset')))['eeg'].values
        band, snr = individual_band_snr(x[:N_SAMPLES_TRAIN], FS)
        method = method_class(band=band, fs=FS, **kwargs)
        y = method.apply(x[:N_SAMPLES_TRAIN+N_SAMPLES_TEST])
        res[j_kwargs, :] = y
    return res, kwargs_list



# Rect env detector
kwargs_grid = {
    'n_taps_bandpass': np.arange(0, DELAY_RANGE.max()+1, np.diff(DELAY_RANGE)[0])*2,
    'n_taps_smooth': np.arange(0, DELAY_RANGE.max()+1, np.diff(DELAY_RANGE)[0])*2
}
res, kwargs_list = grid_apply_method(RectEnvDetector, kwargs_grid)

kwargs_grid = {
    'delay': DELAY_RANGE[::2],
    'n_taps': np.arange(100, 1000 + 1, 200),
    'n_fft': [500, 1000, 2000]
}
res, kwargs_list = grid_apply_method(CFIRBandEnvelopeDetector, kwargs_grid)
