"""
Figure 5: Discrete paradigm accuracy for one subject with median SNR
"""


import pandas as pd
import pylab as plt
import numpy as np
import seaborn as sns
from release.filters import CFIRBandEnvelopeDetector, RectEnvDetector
from release.utils import magnitude_spectrum
from release.constants  import FS, DELAY_RANGE
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score


def get_classes(y, alpha, n_states=3):
    y_pred = np.zeros(len(y))
    if n_states == 3:
        y_pred[y > np.percentile(y, alpha)] = 1
        y_pred[y > np.percentile(y, 100 - alpha)] = 2
    if n_states == 2:
        y_pred[y > np.percentile(y, 100 - alpha)] = 1
    return y_pred

dataset = "alpha2-delay-subj-21_12-06_12-15-09"
eeg_df = pd.read_pickle('data/rest_state_probes_real.pkl').query('dataset=="{}"'.format(dataset))

envelope = eeg_df['an_signal'].abs().values
band = eeg_df[['band_left', 'band_right']].values[0]

magnitude_spectrum_train = {}
_, weights = magnitude_spectrum(eeg_df['eeg'].values, FS)

stats_df = pd.read_pickle('results/stats.pkl').query('dataset=="{}"'.format(dataset))
flatui = {'cfir':'#0099d8', 'acfir': '#84BCDA', 'wcfir':'#FE4A49', 'rect':'#A2A79E'}


alpha=5
#DELAY_RANGE = np.linspace(-50, 100, 51, dtype=int)
acc = np.zeros(len(DELAY_RANGE))
acc_rand = np.zeros(len(DELAY_RANGE))

fig, axes = plt.subplots(2, 2, sharey='col', figsize=(6,6))
plt.subplots_adjust(hspace=0.4, wspace=0.4)

for j_n_states, n_states in enumerate([2, 3]):

    y_true = get_classes(envelope, alpha, n_states)

    for method_name, method_class in zip(
            ['cfir', 'rect', 'wcfir'],
            [CFIRBandEnvelopeDetector, RectEnvDetector, CFIRBandEnvelopeDetector]):
        acc = np.zeros(len(DELAY_RANGE))*np.nan
        for d, DELAY in enumerate(DELAY_RANGE):
            if method_name == 'rect' and DELAY <0: continue
            params = stats_df.query('method=="{}" & metric=="corr" & delay=="{}"'.format(method_name, DELAY*2))['params'].values[0]

            params['weights'] = weights if method_name == 'wcfir' else None
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


            y_pred = get_classes(envelope_pred, alpha, n_states)
            acc[d] = balanced_accuracy_score(y_true, y_pred) if (method_name in ['cfir', 'wcfir'] or DELAY>=0) else np.nan

        axes[j_n_states, 1].plot(DELAY_RANGE*2, acc*100, '.-', label=method_name, color=flatui[method_name])

    axes[j_n_states, 1].plot(DELAY_RANGE*2, DELAY_RANGE*0 + balanced_accuracy_score(y_true, y_true*0)*100, '.-',  color='k', label='all-high')
# [ax.set_xlabel('Delay, ms') for ax in axes[:, 1]]
axes[1, 1].set_xlabel('Delay, ms')
axes[1, 1].legend()
axes[0, 1].set_ylabel('Balanced accuracy score, %')
axes[1, 1].set_ylabel('Balanced accuracy score, %')
axes[0, 0].set_title('A. High/Other\n', x = 0)
axes[1, 0].set_title('B. High/Middle/Low\n', ha='right')
[ax.axvline(0, color='k', linestyle='--', alpha=0.5, zorder=-1000) for ax in axes[:, 1]]
# plt.plot(envelope0ms)
# plt.plot(envelope)
#
# sns.kdeplot(envelope, envelope0ms)

# plt.savefig('results/viz/res-classification.png', dpi=500)

ax = axes
# fig, ax = plt.subplots(2, figsize=(6, 6))
up = np.percentile(envelope*1e6, 100-alpha)
low = np.percentile(envelope*1e6, alpha)
t = np.arange(len(envelope))/500

ax[0, 0].plot(t-58, envelope*1e6, color='k')
ax[0, 0].axhline(np.percentile(envelope*1e6, 100-alpha), color='k', linestyle='--')

ax[0, 0].text(8.5, up+4, 'High', ha='center')
ax[0, 0].text(8.5, up-3, 'Other', ha='center')
# plt.axhspan(np.percentile(envelope*1e6, alpha), np.percentile(envelope*1e6, 100-alpha), color=flatui['cfir'], alpha=0.5)
# plt.axhspan(np.percentile(envelope*1e6, alpha), -1000, color=flatui['wcfir'], alpha=0.5)
ax[0, 0].set_ylim(-7, 20)
ax[0, 0].set_xlim(0, 10)
ax[0, 0].set_ylabel('Envelope, $uV$')

ax[1, 0].plot(t-58, envelope*1e6, color='k')
ax[1, 0].axhline(np.percentile(envelope*1e6, 100-alpha), color='k', linestyle='--')
ax[1, 0].axhline(np.percentile(envelope*1e6, alpha), color='k', linestyle='--')
ax[1, 0].text(8.5, up+4, 'High', ha='center')
ax[1, 0].text(8.5, up-3, 'Middle', ha='center')
ax[1, 0].text(8.5, low-5, 'Low', ha='center')
# plt.axhspan(np.percentile(envelope*1e6, alpha), np.percentile(envelope*1e6, 100-alpha), color=flatui['cfir'], alpha=0.5)
# plt.axhspan(np.percentile(envelope*1e6, alpha), -1000, color=flatui['wcfir'], alpha=0.5)
ax[1, 0].set_ylim(-7, 20)
ax[1, 0].set_xlim(0, 10)
ax[1, 0].set_ylabel('Envelope, $uV$')
ax[1, 0].set_xlabel('Time, s')
plt.savefig('results/viz/res-classification-explained.png', dpi=500)