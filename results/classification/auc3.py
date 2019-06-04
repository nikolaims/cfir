import pandas as pd
import pylab as plt
import numpy as np
import seaborn as sns
from release.filters import CFIRBandEnvelopeDetector, RectEnvDetector
from settings import FS, DELAY_RANGE
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score


def get_classes(y, alpha):
    y_pred = np.zeros(len(y))
    y_pred[y > np.percentile(y, alpha)] = 1
    y_pred[y > np.percentile(y, 100 - alpha)] = 2
    return y_pred

dataset = "alpha2-delay-subj-21_12-06_12-15-09"
eeg_df = pd.read_pickle('data/rest_state_probes.pkl').query('dataset=="{}"'.format(dataset))

envelope = eeg_df['an_signal'].abs().values
band = eeg_df[['band_left_train', 'band_right_train']].values[0]

stats_df = pd.read_pickle('results/stats.pkl').query('dataset=="{}" & delay=={} '.format(dataset, 0))



alpha=10
y_true = get_classes(envelope, alpha)
acc = np.zeros(len(DELAY_RANGE))
acc_rand = np.zeros(len(DELAY_RANGE))

for method_name, method_class in zip(
        ['cfir', 'rect', 'wcfir',],
        [CFIRBandEnvelopeDetector, RectEnvDetector, CFIRBandEnvelopeDetector]):
    for d, DELAY in enumerate(DELAY_RANGE):
        params = stats_df.query('method=="{}" & metric=="corr"'.format(method_name))['params'].values[0]

        env_det = method_class(band=band, fs=FS, delay=DELAY, **params)
        envelope_pred = np.abs(env_det.apply(eeg_df['eeg'].values))

        # params = stats_df.query('method=="rect" & metric=="corr"')['params'].values[0]
        # env_det = WHilbertFilter(band=band, fs=FS, delay=DELAY, **params)
        # envelope_pred = np.abs(env_det.apply(eeg_df['eeg'].values))
        #
        # params = stats_df.query('method=="whilbert" & metric=="corr"')['params'].values[0]
        # env_det = WHilbertFilter(band=band, fs=FS, **params)
        # envelope_pred = np.abs(env_det.apply(eeg_df['eeg'].values))
        #
        # params = stats_df.query('method=="ffiltar" & metric=="corr"')['params'].values[0]
        # env_det = RectEnvDetector(band, FS, params['n_taps'], DELAY)
        # env_det = WHilbertFilter(band=band, fs=FS, **params)


        y_pred = get_classes(envelope_pred, alpha)
        acc[d] = balanced_accuracy_score(y_true, y_pred) if (method_name in ['cfir', 'ffiltar'] or DELAY>=0) else np.nan

    plt.plot(DELAY_RANGE*2, acc*100, label=method_name)


plt.axhline(balanced_accuracy_score(y_true, y_true*0+1)*100, color='k', linestyle='--', label='all-0')
plt.xlabel('Delay, ms')

plt.legend()
plt.ylabel('Balanced accuracy score, %')
plt.axvline(0, color='k', linestyle='--', alpha=0.5)
# plt.plot(envelope0ms)
# plt.plot(envelope)
#
# sns.kdeplot(envelope, envelope0ms)