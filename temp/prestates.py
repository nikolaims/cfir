import numpy as np
import pandas as pd
from release.filters import CFIRBandEnvelopeDetector
from release.constants import FS, N_SAMPLES_TRAIN, N_SAMPLES_TEST
from release.utils import magnitude_spectrum, individual_max_snr_band, delay_align, band_hilbert
import pylab as plt
from scipy.stats import normaltest


# constants
DELAY = 0
TRAIN_SLICE = slice(0, N_SAMPLES_TRAIN)
TEST_SLICE = slice(N_SAMPLES_TRAIN, N_SAMPLES_TRAIN + N_SAMPLES_TEST)


# performance metric
def envelope_corr(pred, target):
    pred, target = delay_align(pred, target, DELAY)
    corr = np.corrcoef(np.abs(pred), np.abs(target))[1, 0]
    return corr


# load data
eeg_df = pd.read_pickle('data/rest_state_probes_real.pkl')
datasets = eeg_df['dataset'].unique()


def get_env(x, band, n_ma = 500):
    x = np.abs(band_hilbert(x, FS, band))
    x = pd.Series(x).rolling(n_ma).mean().dropna().values[::n_ma]
    return x


transitions = np.zeros((15, 5))

for dataset in datasets:

    band = eeg_df.loc[eeg_df.dataset==dataset, 'band_left'].values[0], eeg_df.loc[eeg_df.dataset==dataset, 'band_right'].values[0]
    band = np.array(band)


    x = eeg_df.loc[eeg_df.dataset==dataset, 'eeg']
    x_alpha=get_env(x, band)
    x_beta = get_env(x, band*2)
    x_theta = get_env(x, band/2)
    print(dataset)
    # plt.plot(x)

    thresholds = np.array([np.percentile(x_alpha, 20*k) for k in range(0, 6)])

    for n, x in enumerate([x_alpha, x_beta, x_theta]):
        thresholds_from = np.array([np.percentile(x, 20 * k) for k in range(0, 6)])
        for xnm1, xn in zip(x[:-1], x_alpha[1:]):
            cnm1 = np.where(np.diff(xnm1 > thresholds_from))[0]
            cnm1 = cnm1[0] if len(cnm1)>0 else 4
            cn = np.where(np.diff(xn > thresholds))[0]
            cn = cn[0] if len(cn) > 0 else 4
            transitions[cnm1+5*n, cn] += 1

transitions /= np.sum(transitions, 1)[:, None]

