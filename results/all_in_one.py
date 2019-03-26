import pandas as pd
import pylab as plt
import numpy as np
from settings import FS
from pycfir.filters import CFIRBandEnvelopeDetector

nor = lambda x: x/np.nanmax(np.abs(x))
dataset = "alpha2-delay-subj-21_12-06_12-15-09"
eeg_df = pd.read_pickle('data/rest_state_probes.pkl').query('dataset=="{}"'.format(dataset)).iloc[20000:30000]
eeg = eeg_df['eeg'].values
an_signal = eeg_df['an_signal'].values
band = eeg_df[['band_left_train', 'band_right_train']].values[0]
t0 = 1
slc = slice(6800, 7500)
t = np.arange(slc.stop-slc.start)/FS


fig, axes = plt.subplots(6, 4, sharex=True, figsize=(12,8))
fig.subplots_adjust(hspace=1)
for j, ax in enumerate(axes.flatten()):
    ax.get_yaxis().set_visible(False)
    ax.axvline(t0, color='k', alpha=0.2)
    ax.spines['bottom'].set_edgecolor('#6a747c')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.tick_params(color='#6a747c')

DELAY = 0
stats_df = pd.read_pickle('results/stats.pkl').query('dataset=="{}" & delay=={} '.format(dataset, DELAY))
params = stats_df.query('method=="cfir" & metric=="corr"')['params'].values[0]
rect = CFIRBandEnvelopeDetector(band, FS, DELAY, params['n_taps'])
res = rect.apply(eeg)

filtered = np.nan * t* t.astype('complex')
filtered[(t > t0 - len(rect.b) / FS) & (t <= t0)] = nor(rect.b[::-1])

ax = axes[0, 3]
ax.plot(t, nor(eeg[slc]), '#0099d8')
ax.plot(t, np.real(filtered), 'r', label='$b[-n]$')
ax.plot(t, np.imag(filtered), 'r--')
ax.text(t[0], 0.8, '$x$', color='#0099d8')
ax.text(t[200], 0.8, '$b_{cfir}$', color='r')

axes[1, 3].clear()
axes[1, 3].axis('off')
axes[2, 3].clear()
axes[2, 3].axis('off')

ax = axes[3, 3]
ax.plot(t, np.real(nor(res[slc])), '#0099d8')
ax.plot(t, np.imag(nor(res[slc])), '#0099d8', linestyle='--')
ax.text(t[0], 0.8, '$y_{cfir}$', color='#0099d8')

ax = axes[4, 3]
ax.plot(t, nor(np.abs(res[slc])), '#0099d8')
ax.plot(t, nor(np.abs(an_signal[slc])), 'k', alpha=0.5)
ax.text(t[260], 0.3, '$a_{cfir}$', color='#0099d8')
ax.text(t[200], 0.7, '$a$', color='#444444')
plt.setp(ax.get_xticklabels(), visible=True)
ax.set_xlabel('Time, s')
ax.xaxis.set_label_coords(0.9, -0.1)
ax.tick_params(labelbottom=True)


params = stats_df.query('method=="cfir" & metric=="phase"')['params'].values[0]
rect = CFIRBandEnvelopeDetector(band, FS, DELAY, params['n_taps'])
res = rect.apply(eeg)

ax = axes[5,3]
ax.plot(t, np.angle(res[slc]), '#0099d8')
ax.plot(t, np.angle(an_signal[slc]), 'k', alpha=0.5)
ax.text(t[0], np.pi+0.5, '$\phi_{cfir}$', color='#0099d8')
ax.text(t[-1], np.pi+0.5, '$\phi$', color='#444444')
ax.get_yaxis().set_visible(True)
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])
ax.spines['left'].set_visible(True)
ax.spines['left'].set_edgecolor('#6a747c')
ax.tick_params(labelbottom=False)
#fig.axes[0].set_title('$cfir$ \n $D = {}$ ms'.format(DELAY*2))


fig.savefig('results/viz/all.png', dpi=150)