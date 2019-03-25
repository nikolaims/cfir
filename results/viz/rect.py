import pandas as pd
import pylab as plt
import seaborn as sns
import numpy as np
from settings import FS
import scipy.signal  as sg
from pycfir.filters import RectEnvDetector, WHilbertFilter, band_hilbert, rt_emulate

DELAY = 80
dataset = "alpha2-delay-subj-21_12-06_12-15-09"
eeg_df = pd.read_pickle('data/rest_state_probes.pkl').query('dataset=="{}"'.format(dataset)).iloc[20000:30000]

eeg = eeg_df['eeg'].values
an_signal = eeg_df['an_signal'].values
band = eeg_df[['band_left_train', 'band_right_train']].values[0]
nor = lambda x: x/np.max(np.abs(x))

t0 = 1
slc = slice(6800, 7500)


class RectEnvDetectorViz(RectEnvDetector):
    def apply(self, chunk):
        y1, self.zi_bandpass = sg.lfilter(self.b_bandpass, [1.],  chunk, zi=self.zi_bandpass)
        y2 = np.abs(y1)
        y3, self.zi_smooth  = sg.lfilter(self.b_smooth, [1.], y2, zi=self.zi_smooth)
        return y1, y2, y3

stats_df = pd.read_pickle('results/stats.pkl').query('dataset=="{}" & delay=={} '.format(dataset, DELAY))

rect = RectEnvDetectorViz(band, FS, stats_df.query('method=="rect"')['params'].values[0]['n_taps_bandpass'], DELAY)

steps = [eeg] + list((rect.apply(eeg)))
steps = [x[slc] for x in steps]

fig, axes = plt.subplots(len(steps), sharex=True, figsize=(3, 5))
plt.subplots_adjust(hspace=1)

t = np.arange(slc.stop-slc.start)/FS

steps[-1] = nor(steps[-1])
for j_step, step in enumerate(steps):
    axes[j_step].plot(t, step/np.max(step), '#0099d8')#, alpha=0.5)
    if j_step == 0:
        mask = (t > t0 - len(rect.b_bandpass) / FS) & (t<=t0)

        #axes[j_step].plot(t[mask], step[mask]/np.max(step), 'k', linewidth=2)
        axes[j_step].plot(t[mask], rect.b_bandpass / FS/np.max(step), 'r', linewidth=2, alpha=0.8)
        #axes[j_step].axvspan(1 - len(rect.b_bandpass) / FS, 1, alpha=0.1, color='red')
        axes[j_step].text(t[0], 0.8, '$x$', color='#0099d8')
        axes[j_step].text(t[400], -1.8, '$h_{bp}$', color='r')
    if j_step == 1:
        indexes = np.arange(len(t))[(step>0) | (t>t0)]
        s = step.copy()
        s[indexes] = np.nan
        axes[j_step].plot(t, s/np.max(step), 'k', linewidth=2)
        axes[j_step].text(t[250], -2, r'$\downarrow |\cdot|$', color='k')
    if j_step == 2:
        mask = (t > t0 - len(rect.b_smooth) / FS) & (t <= t0)
        axes[j_step].plot(t[mask], step[mask]/np.max(step), 'k', linewidth=2)

        axes[j_step].plot(t[mask], rect.b_smooth / FS/3/np.max(step), 'r', linewidth=2, alpha=0.8)
        #axes[j_step].axvspan(1 - len(rect.b_smooth) / FS, 1, alpha=0.1, color='red')
        axes[j_step].text(t[510], 1, '$h_{lp}$', color='r')
    if  j_step==3:
        #axes[j_step].plot(t, nor(np.abs(an_signal[slc])), 'k', alpha=0.5)
        axes[j_step].plot(t, nor(np.abs(np.roll(an_signal, DELAY)[slc])), 'k--', alpha=0.5)
        axes[j_step].text(t[350], 0.4, '$a_{rect}$', color='#0099d8')
        #axes[j_step].text(t[150], 0.9, '$a$', color='#444444')
        axes[j_step].text(t[300], 1.1, '$a[n-D]$', color='#777777')


for ax in axes:
    ax.get_yaxis().set_visible(False)
    ax.axvline(t0, color='k', alpha=0.2)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.spines['bottom'].set_edgecolor('#6a747c')
    ax.tick_params(color='#6a747c')



axes[-1].set_xlabel('Time, s')
axes[-1].xaxis.set_label_coords(0.9, -0.15)

axes[0].set_title('$rect$ \n $D = {}$ ms'.format(DELAY*2))
fig.savefig('results/viz/rect.png', dpi=200)