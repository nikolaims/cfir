import pandas as pd
import pylab as plt
import numpy as np
import seaborn as sns
from pycfir.filters import CFIRBandEnvelopeDetector
from settings import FS, DELAY_RANGE
from sklearn.metrics import roc_auc_score, average_precision_score

def get_thresholds(x, alpha):
    high_threshold = np.percentile(x, 100-alpha)
    low_threshold = np.percentile(x, alpha)
    return {'low': low_threshold, 'high': high_threshold}

dataset = "alpha2-delay-subj-21_12-06_12-15-09"
eeg_df = pd.read_pickle('data/rest_state_probes.pkl').query('dataset=="{}"'.format(dataset))

envelope = eeg_df['an_signal'].abs().values
thresholds = get_thresholds(envelope, 10)
band = eeg_df[['band_left_train', 'band_right_train']].values[0]

stats_df = pd.read_pickle('results/stats.pkl').query('dataset=="{}" & delay=={} '.format(dataset, 50))



DELAY_RANGE = np.arange(-50, 200, 10)
for f, c in zip(np.linspace(0.5, 1.5, 5), sns.color_palette()):

    tprs = []
    roc = []
    for DELAY in DELAY_RANGE:
        params = stats_df.query('method=="cfir" & metric=="corr"')['params'].values[0]
        env_det = CFIRBandEnvelopeDetector(band, FS, DELAY, params['n_taps'])
        envelope0ms = np.abs(env_det.apply(eeg_df['eeg'].values))
        if f<1:
            tprs.append(average_precision_score(envelope < f*np.median(envelope), 1-envelope0ms/envelope0ms.max()))
            roc.append(roc_auc_score(envelope < f*np.median(envelope), 1-envelope0ms/envelope0ms.max()))
        else:
            tprs.append(average_precision_score(envelope > f * np.median(envelope), envelope0ms / envelope0ms.max()))
            roc.append(roc_auc_score(envelope > f * np.median(envelope), envelope0ms / envelope0ms.max()))

    plt.plot(DELAY_RANGE*2, np.array(roc)*100, color=c, label='{}{}'.format('roc', f))
    plt.plot(DELAY_RANGE*2, np.array(tprs)*100, '--', color=c, label='{}{}'.format('rec', f))
    plt.xlabel('Delay, ms')
    plt.ylabel('# detections / # spindles [%]')
    plt.axvline(0, color='k', linestyle='--')

plt.legend()
# plt.plot(envelope0ms)
# plt.plot(envelope)
#
# sns.kdeplot(envelope, envelope0ms)