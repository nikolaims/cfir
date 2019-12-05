import numpy as np
import pandas as pd
from release.filters import CFIRBandEnvelopeDetector
from release.constants import FS, N_SAMPLES_TRAIN, N_SAMPLES_TEST
from release.utils import magnitude_spectrum, individual_max_snr_band, delay_align


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

# loop over datasets
res = np.empty((len(datasets), N_SAMPLES_TRAIN + N_SAMPLES_TEST), dtype='complex64')
test_scores = np.zeros(len(datasets))
for j_dataset, dataset in enumerate(datasets):

    # load input signal 'x' and target signal 'target' for specific dataset (subject)
    df = eeg_df.query('dataset=="{}"'.format(dataset))
    x = df['eeg'].values
    target = df['an_signal'].values

    # train data: define 'train_band', 'train_weights' and 'train_n_taps' parameters on train data
    x_train = x[TRAIN_SLICE]
    train_band, _snr_train = individual_max_snr_band(x_train, FS)
    _freqs, train_weights = magnitude_spectrum(x_train, FS)
    best_train_score = 0
    train_n_taps = None
    for n_taps in [250, 500, 750, 1000]:
        method = CFIRBandEnvelopeDetector(band=train_band, fs=FS, delay=DELAY, n_taps=n_taps, weights=train_weights)
        res[j_dataset, TRAIN_SLICE] = method.apply(x_train)
        train_score = envelope_corr(res[j_dataset, TRAIN_SLICE], target[TRAIN_SLICE])
        if train_score > best_train_score:
            train_n_taps = n_taps
            best_train_score = train_score

    # test data: estimate performance metric on test data
    method = CFIRBandEnvelopeDetector(band=train_band, fs=FS, delay=DELAY, n_taps=train_n_taps, weights=train_weights)
    x_test = x[TEST_SLICE]
    res[j_dataset, TEST_SLICE] = method.apply(x_test)
    test_scores[j_dataset] = envelope_corr(res[j_dataset, TEST_SLICE], target[TEST_SLICE])

# print mean and std of test scores
print('test score: {:.4f} +- {:.4f}'.format(test_scores.mean(), test_scores.std()))