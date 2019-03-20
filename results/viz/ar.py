import pandas as pd
import pylab as plt
import seaborn as sns
import numpy as np
from statsmodels.regression import yule_walker

from settings import FS
import scipy.signal  as sg
from pycfir.filters import FiltFiltARHilbertFilter, band_hilbert, rt_emulate

DELAY = 0
eeg_df = pd.read_pickle('data/rest_state_probes.pkl')

eeg = eeg_df.query('dataset=="alpha2-delay-subj-12_11-21_20-23-29"')['eeg'].values[:10000]
an_signal = eeg_df.query('dataset=="alpha2-delay-subj-12_11-21_20-23-29"')['an_signal'].values[:10000]
band = eeg_df.query('dataset=="alpha2-delay-subj-12_11-21_20-23-29"')[['band_left_train', 'band_right_train']].values[0]
#band = (8, 12)



class FiltFiltARHilbertFilterViz(FiltFiltARHilbertFilter):
    def apply(self, chunk):
        if chunk.shape[0] <= self.max_chunk_size:
            x = self.buffer.update_buffer(chunk)
            y = sg.filtfilt(*self.ba_bandpass, x)
            if self.delay < self.n_taps_edge_left:

                ar, s = yule_walker(y.real[:-self.n_taps_edge_left], self.ar_order, 'mle')

                pred = y.real[:-self.n_taps_edge_left].tolist()
                for _ in range(self.n_taps_edge_left + self.n_taps_edge_right):
                    pred.append(ar[::-1].dot(pred[-self.ar_order:]))

                an_signal = sg.hilbert(pred)
                env = an_signal[-self.n_taps_edge_right-self.delay-len(chunk)+1:-self.n_taps_edge_right-self.delay+1]*np.ones(len(chunk))

                #
                # plt.plot(x, alpha=0.1)
                # plt.plot(pred, alpha=0.9)
                # plt.plot(np.abs(an_signal))
                # plt.plot(y[:-self.n_taps_edge_left], 'k')
                # plt.plot(y, 'k--')
                #
                # plt.show()


            else:
                env = sg.hilbert(y)[-self.delay-len(chunk)+1:-self.delay+1] * np.ones(len(chunk))

            return env, y, pred

        else:
            return rt_emulate(self, chunk, self.max_chunk_size)


stats_df = pd.read_pickle('results/stats.pkl').query('dataset=="alpha2-delay-subj-12_11-21_20-23-29" & delay=={} & metric=="corr"'.format(DELAY))

params = stats_df.query('method=="ffiltar"')['params'].values[0]
rect = FiltFiltARHilbertFilterViz(band, FS, params['n_taps_edge'], DELAY, params['ar_order'], 1, buffer_s=params['buffer_s'])

res = rect.apply(eeg)


t0 = 0.8
slc = slice(6600, 6600+int(FS*1))
t = np.arange(slc.stop-slc.start)/FS


filtered = np.nan * t
filtered[(t > t0 - len(rect.buffer.buffer) / FS) & (t <= t0)] = res[1::3][slc][t <= t0][-1]
predicted = np.nan * t
predicted[(t > t0 - params['n_taps_edge'] / FS) & (t <= t0 + params['n_taps_edge']/FS)] = res[2::3][slc][t <= t0][-1][-2*params['n_taps_edge']:]

full_predicted = np.nan * t.astype('complex')
full_predicted[(t > t0 - len(rect.buffer.buffer) / FS) & (t <= t0 + params['n_taps_edge']/FS)] = sg.hilbert(res[2::3][slc][t <= t0][-1])
nor = lambda x: x/np.max(np.abs(x))


fig = plt.figure(figsize=(3, 8))
ax0 = fig.add_subplot(6,1,1)
ax0.plot(t, eeg[slc], '#0099d8')

ax0.plot(t[(t>t0-len(rect.buffer.buffer)/FS) & (t<=t0)], eeg[slc][(t>t0-len(rect.buffer.buffer)/FS) & (t<=t0)], 'k', linewidth=2)



ax = fig.add_subplot(6,1,2, sharex=ax0)

ax.plot(t, filtered, '#0099d8')

ax.plot(t, predicted, 'r--')
#ax.plot(t0-DELAY/FS, filtered[t <= (t0-DELAY/FS)][-1], 'or')

ax = fig.add_subplot(6,1,3, sharex=ax0)
ax.plot(t, np.real(full_predicted), '#0099d8')
ax.plot(t, np.imag(full_predicted), '#0099d8', linestyle='--')
ax.plot(t0, np.real(full_predicted)[t <= (t0)][-1], 'or')

ax = fig.add_subplot(6,1,4, sharex=ax0)
step = np.concatenate(res[::3])[slc]
ax.plot(t, np.real(step), '#0099d8')
ax.plot(t, np.imag(step), '#0099d8', linestyle='--')

ax = fig.add_subplot(6,1,5, sharex=ax0)
ax.plot(t, nor(np.abs(step)), '#0099d8')
ax.plot(t, nor(np.abs(an_signal[slc])), 'k', alpha=0.5)
ax.plot(t, nor(np.abs(np.roll(an_signal, DELAY)[slc])), 'k--', alpha=0.5)

ax = fig.add_subplot(6,1,6, sharex=ax0)
ax.plot(t, np.angle(step), '#0099d8')
ax.plot(t, np.angle(an_signal[slc]), 'k', alpha=0.5)
ax.plot(t, np.angle(np.roll(an_signal, DELAY)[slc]), 'k--', alpha=0.5)


for j, ax in enumerate(fig.axes):
    ax.get_yaxis().set_visible(False)
    ax.axvline(t0, color='k', alpha=0.2)
    ax.spines['bottom'].set_edgecolor('#6a747c')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.tick_params(color='#6a747c')
    if j not in [4]:
        plt.setp(ax.get_xticklabels(), visible=False)
    if j == 4:
        ax.set_xlabel('Time, s')
fig.subplots_adjust(hspace=1)

fig.savefig('results/viz/ffiltar.png', dpi=150)