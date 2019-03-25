import pandas as pd
import pylab as plt
import seaborn as sns
import numpy as np
from settings import FS
import scipy.signal  as sg
from pycfir.filters import RectEnvDetector, WHilbertFilter, band_hilbert, rt_emulate

DELAY = 40
dataset = "alpha2-delay-subj-21_12-06_12-15-09"
eeg_df = pd.read_pickle('data/rest_state_probes.pkl').query('dataset=="{}"'.format(dataset)).iloc[20000:30000]

eeg = eeg_df['eeg'].values
an_signal = eeg_df['an_signal'].values
band = eeg_df[['band_left_train', 'band_right_train']].values[0]
nor = lambda x: x/np.max(np.abs(x))

t0 = 1
slc = slice(6800, 7500)


stats_df = pd.read_pickle('results/stats.pkl').query('dataset=="{}" & delay=={} '.format(dataset, DELAY))

params = stats_df.query('method=="whilbert" & metric=="corr"')['params'].values[0]


class WHilbertFilterViz(WHilbertFilter):

    def apply(self, chunk):
        if chunk.shape[0] < self.buffer.n_taps:
            x = self.buffer.update_buffer(chunk)
            Xf = np.fft.fft(x, None)
            Xf1 = Xf.copy()
            w = np.fft.fftfreq(x.shape[0], d=1. / FS)
            Xf[(w < band[0]) | (w > band[1])] = 0
            y = 2*np.fft.ifft(Xf)
            return y[-chunk.shape[0]-self.delay-1:-self.delay-1], y, Xf1, Xf
        else:
            return rt_emulate(self, chunk, self.max_chunk_size)


rect = WHilbertFilterViz(stats_df.query('method=="whilbert"')['params'].values[0]['n_taps'], FS, band, DELAY, max_chunk_size=1)

res = rect.apply(eeg)


t = np.arange(slc.stop-slc.start)/FS


hilb = np.nan*t.astype('complex')
hilb[(t>t0-len(rect.buffer.buffer)/FS) & (t<=t0)] = res[1::4][slc][t<=t0][-1]

steps = [eeg, hilb, np.concatenate(res[::4]), np.abs(np.concatenate(res[::4]))]

steps = [x[slc] if j!=1 else x for j, x in enumerate(steps)]
nor = lambda x: x/np.nanmax(np.abs(x))


fig = plt.figure(figsize=(3, 8))
ax0 = fig.add_subplot(6,1,1)
ax0.plot(t, nor(eeg[slc]), '#0099d8')
ax0.plot(t[(t>t0-len(rect.buffer.buffer)/FS) & (t<=t0)], nor(eeg[slc])[(t>t0-len(rect.buffer.buffer)/FS) & (t<=t0)], 'k', linewidth=2)
ax0.text(t[0], 0.8, '$x$', color='#0099d8')
ax0.text(t[250], -2, r'$\downarrow W$', color='k')

ax = fig.add_subplot(6,1,2)
Xf = np.fft.fftshift(np.abs(res[2::4][slc][t<=t0][-1]))
w = np.fft.fftshift(np.fft.fftfreq(len(Xf), d=1. / FS))
Xf1 = Xf.copy()
Xf1[(w < band[0]) | (w > band[1])]  = np.nan
ax.plot(w, Xf, 'k', alpha=0.3)
ax.fill_between(w, Xf1, color='r')
ax.plot([-FS/2, FS/2], [0,0], color='r')
ax.set_xlim(-590, 490)
ax.set_xticks([-250, 0, 10, 250])
ax.set_xticklabels(['$-0.5f_s$', '0  ', '   $f_0$', '$0.5f_s$'])


ax.spines['bottom'].set_edgecolor('w')

ax = fig.add_subplot(6,1,3, sharex=ax0)
step=nor(hilb)
ax.plot(t, np.real(step), '#0099d8')
ax.plot(t, np.imag(step), '#0099d8', linestyle='--')
ax.plot(t0-DELAY/FS, step[t <= (t0-DELAY/FS)][-1], 'or')
ax.plot(t0, step[t <= (t0 - DELAY / FS)][-1], 'or', fillstyle='none')
ax.text(t[250], 1.6, r'$\downarrow W^H$', color='k')
ax.text(t[500], 1, '$y_{hilb}$', color='r')

ax = fig.add_subplot(6,1,4, sharex=ax0)
step = np.concatenate(res[::4])[slc]
ax.plot(t, np.real(nor(step)), '#0099d8')
ax.plot(t, np.imag(nor(step)), '#0099d8', linestyle='--')
ax.text(t[0], 0.8, '$y_{hilb}$', color='#0099d8')

ax = fig.add_subplot(6,1,5, sharex=ax0)
ax.plot(t, nor(np.abs(step)), '#0099d8')
#ax.plot(t, nor(np.abs(an_signal[slc])), 'k', alpha=0.5)
ax.plot(t, nor(np.abs(np.roll(an_signal, DELAY)[slc])), 'k--', alpha=0.5)
ax.text(t[150], 0.5, '$a_{hilb}$', color='#0099d8')
ax.text(t[290], 1.1, '$a[n-D]$', color='#777777')
#ax.text(t[150], 0.9, '$a$', color='#444444')


ax = fig.add_subplot(6,1,6, sharex=ax0)
step = np.angle(step)
ax.plot(t, step, '#0099d8')
#ax.plot(t, np.angle(an_signal[slc]), 'k', alpha=0.5)
ax.plot(t, np.angle(np.roll(an_signal, DELAY)[slc]), 'k--', alpha=0.5)
ax.text(t[0], np.pi+0.5, '$\phi_{hilb}$', color='#0099d8')
ax.text(t[50], -np.pi-2, '$\phi[n-D]$', color='#444444')


for j, ax in enumerate(fig.axes):
    ax.get_yaxis().set_visible(False)
    if j!=1:
        ax.axvline(t0, color='k', alpha=0.2)
        ax.spines['bottom'].set_edgecolor('#6a747c')
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_position('zero')
    ax.tick_params(color='#6a747c')
    if j not in [1, 4]:
        plt.setp(ax.get_xticklabels(), visible=False)
    if j == 4:
        ax.set_xlabel('Time, s')
        ax.xaxis.set_label_coords(0.9, -0.15)


ax.get_yaxis().set_visible(True)
ax.set_yticks([-np.pi, 0, np.pi])
ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])
ax.spines['left'].set_visible(True)
ax.spines['left'].set_edgecolor('#6a747c')

fig.axes[0].set_title('$hilb$ \n $D = {}$ ms'.format(DELAY*2))

fig.subplots_adjust(hspace=1.2)

fig.savefig('results/viz/whilb.png', dpi=200)