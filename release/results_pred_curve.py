import pandas as pd
import pylab as plt
import numpy as np
import seaborn as sns
from release.filters import CFIRBandEnvelopeDetector, RectEnvDetector
from release.utils import magnitude_spectrum
from release.constants import FS, DELAY_RANGE
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score


def get_classes(y, alpha, n_states=3):
    y_pred = np.zeros(len(y))
    if n_states == 3:
        y_pred[y > np.percentile(y, alpha)] = 1
        y_pred[y > np.percentile(y, 100 - alpha)] = 2
    if n_states == 2:
        y_pred[y > np.percentile(y, 100 - alpha)] = 1
    return y_pred

eeg_df = pd.read_pickle('data/rest_state_probes_real.pkl')
dataset = eeg_df.query('snr>0.9')['dataset'].unique()[0]
eeg_df = eeg_df.query('dataset=="{}"'.format(dataset))

envelope = eeg_df['an_signal'].abs().values * 1e6
band = eeg_df[['band_left', 'band_right']].values[0]

magnitude_spectrum_train = {}
_, weights = magnitude_spectrum(eeg_df['eeg'].values*1e6, FS)

stats_df = pd.read_pickle('results/stats.pkl').query('dataset=="{}"'.format(dataset))
flatui = {'cfir':'#0099d8', 'acfir': '#84BCDA', 'wcfir':'#FE4A49', 'rect':'#A2A79E'}


DELAY_RANGE = np.array([-50, 0, 50, 150])
acc = np.zeros(len(DELAY_RANGE))
acc_rand = np.zeros(len(DELAY_RANGE))

fig, axes = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(7,3))
plt.subplots_adjust(bottom=0.17, left=0.1, wspace=0, hspace=0)



pre = 375
post = 250
t = np.arange(-pre, post)/FS*1000


delay_dict = {-150: -50, -50: -50, -25:-25,   0:0,  25:25,  50:50, 100:50, 150:50}
lines = []
for method_name, method_class in zip(
        ['cfir', 'rect', 'wcfir'][1:3],
        [CFIRBandEnvelopeDetector, RectEnvDetector, CFIRBandEnvelopeDetector][1:3]):
    for d, DELAY in enumerate(DELAY_RANGE[::-1]):
        if method_name == 'rect' and DELAY < 0: continue
        params = stats_df.query('method=="{}" & metric=="corr" & delay=="{}"'.format(method_name, delay_dict[DELAY]))['params'].values[0]

        params['weights'] = weights if method_name == 'wcfir' else None
        env_det = method_class(band=band, fs=FS, delay=DELAY, **params)
        envelope_pred = np.abs(env_det.apply(eeg_df['eeg'].values*1e6))
        detections = np.where(np.diff(1*(envelope_pred>np.percentile(envelope_pred, 95)))>0)[0]
        spindles = []
        for detection in detections:
            if detection > pre and detection < len(envelope_pred) - post:
                spindles.append(envelope[detection-pre:detection+post])
        h, = axes[d//3, d%3].plot(t, np.mean(spindles, 0), color=flatui[method_name])
        # axes[d // 3, d % 3].plot(t, np.array(spindles).T, color=flatui[method_name], alpha=0.5)
        axes[d // 3, d % 3].fill_between(t, np.mean(spindles, 0)-np.std(spindles, 0), np.mean(spindles, 0)+np.std(spindles, 0), color=flatui[method_name], alpha=0.1)
        axes[d // 3, d % 3].text(-740, 12, '{}ms'.format(DELAY*2))
        if d==0: lines.append(h)

spindles = []
for detection in np.random.randint(pre, len(envelope)-post, 1000):
    spindles.append(envelope[detection-pre:detection+post])
h, = axes[1, 1].plot(t, np.mean(spindles, 0), color='k')
lines.append(h)
axes[1, 1].fill_between(t, np.mean(spindles, 0) - np.std(spindles, 0),
                                 np.mean(spindles, 0) + np.std(spindles, 0), color='k', alpha=0.1)
axes[1, 1].text(-740, 12, 'rand')
[ax.axvline(0, linestyle='--', color='k', alpha=0.5, zorder=-100) for ax in axes.ravel()]


detections = np.where(np.diff(1*(envelope>np.percentile(envelope, 95)))>0)[0]
spindles = []
for detection in detections:
    if detection > pre and detection < len(envelope) - post:
        spindles.append(envelope[detection-pre:detection+post])
h, = axes[1, 2].plot(t, np.mean(spindles, 0), 'k--')
# axes[d // 3, d % 3].plot(t, np.array(spindles).T, color=flatui[method_name], alpha=0.5)
axes[1, 2].fill_between(t, np.mean(spindles, 0)-np.std(spindles, 0), np.mean(spindles, 0)+np.std(spindles, 0), color='k', alpha=0.1)
axes[1, 2].text(-740, 12, 'ideal')
lines.append(h)


axes[1,1].set_xticks([-500, 0, 500])
fig.text(0.02, 0.5, 'Envelope, $\mu V$', va='center', rotation='vertical')
fig.text(0.5, 0.04, 'Time, ms', ha='center')

plt.figlegend(lines, ['rect', 'wcfir', 'rand', 'ideal'])
plt.savefig('results/viz/res-spindles.png', dpi=500)