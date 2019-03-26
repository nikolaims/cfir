import pandas as pd
import pylab as plt
import numpy as np
import scipy.signal as sg
from statsmodels.regression import yule_walker

from settings import FS
from pycfir.filters import CFIRBandEnvelopeDetector, WHilbertFilter, RectEnvDetector, FiltFiltARHilbertFilter, rt_emulate

nor = lambda x: x/np.nanmax(np.abs(x))


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


class RectEnvDetectorViz(RectEnvDetector):
    def apply(self, chunk):
        y1, self.zi_bandpass = sg.lfilter(self.b_bandpass, [1.],  chunk, zi=self.zi_bandpass)
        y2 = np.abs(y1)
        y3, self.zi_smooth  = sg.lfilter(self.b_smooth, [1.], y2, zi=self.zi_smooth)
        return y1, y2, y3



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
            else:
                env = sg.hilbert(y)[-self.delay-len(chunk)+1:-self.delay+1] * np.ones(len(chunk))
            return env, y, pred
        else:
            return rt_emulate(self, chunk, self.max_chunk_size)

dataset = "alpha2-delay-subj-21_12-06_12-15-09"
eeg_df = pd.read_pickle('data/rest_state_probes.pkl').query('dataset=="{}"'.format(dataset)).iloc[20000:30000]
eeg = eeg_df['eeg'].values
an_signal = eeg_df['an_signal'].values
band = eeg_df[['band_left_train', 'band_right_train']].values[0]
t0 = 1
slc = slice(6800, 7500)
t = np.arange(slc.stop-slc.start)/FS


fig, axes = plt.subplots(6, 4, figsize=(12,8))
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
    plt.setp(ax.get_xticklabels(), visible=False)
    ax.set_xlim(0, t.max())
    ax.set_ylim(-1.1, 1.1)

for j, ax in enumerate(axes[5]):
    ax.get_yaxis().set_visible(True)
    ax.set_yticks([-np.pi, 0, np.pi])
    ax.set_yticklabels(['$-\pi$', '0', '$\pi$'])
    ax.spines['left'].set_visible(True)
    ax.spines['left'].set_edgecolor('#6a747c')

for j, ax in enumerate(axes[4]):
    plt.setp(ax.get_xticklabels(), visible=True)
    ax.get_xaxis().set_visible(True)
    ax.tick_params(labelbottom=True)
    ax.set_xticks([0,  1])
    ax.set_xticklabels(['$t_0-1s$',  '$t_0$'], ha = 'left')
    ax.set_ylim(0, 1.1)
    # if j == len(axes[4])-1:
    #     ax.set_xlabel('Time, s')
    #     ax.xaxis.set_label_coords(0.9, -0.1)


DELAY = 0
stats_df = pd.read_pickle('results/stats.pkl').query('dataset=="{}" & delay=={} '.format(dataset, DELAY))
params = stats_df.query('method=="cfir" & metric=="corr"')['params'].values[0]
rect = CFIRBandEnvelopeDetector(band, FS, DELAY, params['n_taps'])
res = rect.apply(eeg)

filtered = np.nan * t* t.astype('complex')
filtered[(t > t0 - len(rect.b) / FS) & (t <= t0)] = nor(rect.b[::-1])

axes[0, 3].set_title('$cfir$ \n $D = {}$ ms'.format(DELAY*2))

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



params = stats_df.query('method=="cfir" & metric=="phase"')['params'].values[0]
rect = CFIRBandEnvelopeDetector(band, FS, DELAY, params['n_taps'])
res = rect.apply(eeg)

ax = axes[5,3]
ax.plot(t, np.angle(res[slc]), '#0099d8')
ax.plot(t, np.angle(an_signal[slc]), 'k', alpha=0.5)
ax.text(t[0], np.pi+0.5, '$\phi_{cfir}$', color='#0099d8')
ax.text(t[-1], np.pi+0.5, '$\phi$', color='#444444')




############################################

DELAY = 40
stats_df = pd.read_pickle('results/stats.pkl').query('dataset=="{}" & delay=={} '.format(dataset, DELAY))
params = stats_df.query('method=="whilbert" & metric=="corr"')['params'].values[0]
rect = WHilbertFilterViz(stats_df.query('method=="whilbert"')['params'].values[0]['n_taps'], FS, band, DELAY, max_chunk_size=1)
res = rect.apply(eeg)
hilb = np.nan*t.astype('complex')
hilb[(t>t0-len(rect.buffer.buffer)/FS) & (t<=t0)] = res[1::4][slc][t<=t0][-1]
steps = [eeg, hilb, np.concatenate(res[::4]), np.abs(np.concatenate(res[::4]))]
steps = [x[slc] if j!=1 else x for j, x in enumerate(steps)]

axes[0, 1].set_title('$hilb$ \n $D = {}$ ms'.format(DELAY*2))

ax0 = axes[0,1]
ax0.plot(t, nor(eeg[slc]), '#0099d8')
ax0.text(t[0], 0.8, '$x$', color='#0099d8')
ax0.text(t[250], -2, r'$\downarrow W$', color='k')
ax0.plot([t0, t0-len(rect.buffer.buffer)/FS, t0-len(rect.buffer.buffer)/FS, t0, t0], [-1,-1, 1, 1, -1], color='r', linewidth=1, alpha=0.5)

ax = axes[1,1]
Xf = np.fft.fftshift(np.abs(res[2::4][slc][t<=t0][-1]))
w = np.fft.fftshift(np.fft.fftfreq(len(Xf), d=1. / FS))
Xf1 = Xf.copy()
Xf1[(w < band[0]) | (w > band[1])]  = 0
ax.plot(w, Xf, '#0099d8', alpha=1)
ax.plot(w, Xf1, color='r')
#ax.plot([-FS/2, FS/2], [0,0], color='r')
ax.set_xlim(-60, 60)
ax.set_ylim(0, max(Xf)*1.05)
ax.set_xticks([-250, 0, 10, 250])
ax.set_xticklabels(['$-0.5f_s$', '0  ', '   $f_0$', '$0.5f_s$'])
ax.spines['bottom'].set_edgecolor('w')
plt.setp(ax.get_xticklabels(), visible=True)
ax.get_xaxis().set_visible(True)
ax.tick_params(labelbottom=True)

ax = axes[2,1]
step=nor(hilb)
ax.plot(t, np.real(step), '#0099d8')
ax.plot(t, np.imag(step), '#0099d8', linestyle='--')
ax.plot(t0-DELAY/FS, step[t <= (t0-DELAY/FS)][-1], 'or')
ax.plot(t0, step[t <= (t0 - DELAY / FS)][-1], 'or', fillstyle='none')
ax.text(t[250], 1.6, r'$\downarrow W^H$', color='k')
ax.text(t[500], 1, '$y_{hilb}$', color='r')

ax = axes[3,1]
step = np.concatenate(res[::4])[slc]
ax.plot(t, np.real(nor(step)), '#0099d8')
ax.plot(t, np.imag(nor(step)), '#0099d8', linestyle='--')
ax.text(t[0], 0.8, '$y_{hilb}$', color='#0099d8')

ax = axes[4,1]
ax.plot(t, nor(np.abs(step)), '#0099d8')
ax.plot(t, nor(np.abs(np.roll(an_signal, DELAY)[slc])), 'k--', alpha=0.5)
ax.text(t[150], 0.5, '$a_{hilb}$', color='#0099d8')
ax.text(t[290], 1.1, '$a[n-D]$', color='#777777')

ax = axes[5,1]
step = np.angle(step)
ax.plot(t, step, '#0099d8')
ax.plot(t, np.angle(np.roll(an_signal, DELAY)[slc]), 'k--', alpha=0.5)
ax.text(t[0], np.pi+0.5, '$\phi_{hilb}$', color='#0099d8')
ax.text(t[50], -np.pi-2, '$\phi[n-D]$', color='#444444')


############################################
DELAY = 80
stats_df = pd.read_pickle('results/stats.pkl').query('dataset=="{}" & delay=={} '.format(dataset, DELAY))

rect = RectEnvDetectorViz(band, FS, stats_df.query('method=="rect"')['params'].values[0]['n_taps_bandpass'], DELAY)

steps = [eeg] + list((rect.apply(eeg)))
steps = [x[slc] for x in steps]
steps[-1] = nor(steps[-1])

axes[0, 0].set_title('$rect$ \n $D = {}$ ms'.format(DELAY*2))

ax = axes[0, 0]
step = steps[0]
ax.plot(t, nor(step), '#0099d8')
mask = (t > t0 - len(rect.b_bandpass) / FS) & (t<=t0)
ax.plot(t[mask], rect.b_bandpass/np.max(rect.b_bandpass), 'r', linewidth=2, alpha=0.8)
ax.text(t[0], 0.8, '$x$', color='#0099d8')
ax.text(t[400], -1.8, '$h_{bp}$', color='r')

ax = axes[1, 0]
step = steps[1]
indexes = np.arange(len(t))[(step>0) | (t>t0)]
s = step.copy()
s[indexes] = np.nan
ax.plot(t, step/np.max(step), '#0099d8')
#ax.plot(t, s/np.max(step), 'k', linewidth=2)
ax.text(t[250], -2, r'$\downarrow |\cdot|$', color='k')

ax = axes[2, 0]
step = steps[2]
mask = (t > t0 - len(rect.b_smooth) / FS) & (t <= t0)
ax.plot(t, step/np.max(step), '#0099d8')
#ax.plot(t[mask], step[mask]/np.max(step), 'k', linewidth=2)
ax.plot(t[mask], rect.b_smooth /np.max(rect.b_smooth), 'r', linewidth=2, alpha=0.8)
ax.text(t[510], 1, '$h_{lp}$', color='r')
ax.set_ylim(0, 1)

axes[3, 0].clear()
axes[3, 0].axis('off')

ax = axes[4, 0]
step = steps[3]
ax.plot(t, step/np.max(step), '#0099d8')
ax.plot(t, nor(np.abs(np.roll(an_signal, DELAY)[slc])), 'k--', alpha=0.5)
ax.text(t[350], 0.4, '$a_{rect}$', color='#0099d8')
ax.text(t[300], 1.1, '$a[n-D]$', color='#777777')

axes[5, 0].clear()
axes[5, 0].axis('off')


##################################################################
DELAY = 0
stats_df = pd.read_pickle('results/stats.pkl').query('dataset=="{}" & delay=={} '.format(dataset, DELAY))
params = stats_df.query('method=="ffiltar" & metric=="corr"')['params'].values[0]
rect = FiltFiltARHilbertFilterViz(band, FS, params['n_taps_edge'], DELAY, params['ar_order'], 1, buffer_s=params['buffer_s'])
res = rect.apply(eeg)

filtered = np.nan * t
filtered[(t > t0 - len(rect.buffer.buffer) / FS) & (t <= t0)] = res[1::3][slc][t <= t0][-1]
predicted = np.nan * t
predicted[(t > t0 - params['n_taps_edge'] / FS) & (t <= t0 + params['n_taps_edge']/FS)] = res[2::3][slc][t <= t0][-1][-2*params['n_taps_edge']:]
full_predicted = np.nan * t.astype('complex')
full_predicted[(t > t0 - len(rect.buffer.buffer) / FS) & (t <= t0 + params['n_taps_edge']/FS)] = sg.hilbert(res[2::3][slc][t <= t0][-1])

axes[0, 2].set_title('$arhilb$ \n $D = {}$ ms'.format(DELAY*2))

ax0 = axes[0, 2]
ax0.plot(t, nor(eeg[slc]), '#0099d8')
ax0.text(t[30], 0.5, '$x$', color='#0099d8')
ax0.plot([t0, t0-len(rect.buffer.buffer)/FS, t0-len(rect.buffer.buffer)/FS, t0, t0], [-1,-1, 1, 1, -1], color='r', linewidth=1, alpha=0.5)

ax = axes[1, 2]
ax.plot(t, filtered/np.nanmax(filtered), '#0099d8')
ax.plot(t, predicted/np.nanmax(filtered), 'r-')
ax.text(t[140], 0.8, r'$\tilde x$', color='#0099d8')
ax.text(t[440], 0.8, r'$\mathcal{P}_{AR(p)}$', color='r')
ax.text(t[440], 0.4, r'$\longrightarrow$', color='r')
ax.text(t[250], -2, r'$\downarrow h$', color='k')
#ax.plot(t0-DELAY/FS, filtered[t <= (t0-DELAY/FS)][-1], 'or')

ax = axes[2, 2]
ax.plot(t, np.real(full_predicted)/np.nanmax(np.real(full_predicted)), '#0099d8')
ax.plot(t, np.imag(full_predicted)/np.nanmax(np.real(full_predicted)), '#0099d8', linestyle='--')
ax.plot(t0, np.real(full_predicted)[t <= (t0)][-1]/np.nanmax(np.real(full_predicted)), 'or')
ax.text(t[440], 0.8, r'$y_{arhilb}$', color='r')

ax = axes[3, 2]
step = np.concatenate(res[::3])[slc]
ax.plot(t, np.real(nor(step)), '#0099d8')
ax.plot(t, np.imag(nor(step)), '#0099d8', linestyle='--')
ax.text(t[0], 0.8, '$y_{arhilb}$', color='#0099d8')

ax = axes[4, 2]
ax.plot(t, nor(np.abs(step)), '#0099d8')
ax.plot(t, nor(np.abs(an_signal[slc])), 'k', alpha=0.5)
#ax.plot(t, nor(np.abs(np.roll(an_signal, DELAY)[slc])), 'k--', alpha=0.5)
ax.text(t[290], 0.2, '$a_{arhilb}$', color='#0099d8')
ax.text(t[240], 0.9, '$a$', color='#444444')

params = stats_df.query('method=="ffiltar" & metric=="phase"')['params'].values[0]
rect = FiltFiltARHilbertFilterViz(band, FS, params['n_taps_edge'], DELAY, params['ar_order'], 1, buffer_s=params['buffer_s'])
res = rect.apply(eeg)
step = np.concatenate(res[::3])[slc]

ax = axes[5, 2]
ax.plot(t, np.angle(step), '#0099d8')
ax.plot(t, np.angle(an_signal[slc]), 'k', alpha=0.5)
#ax.plot(t, np.angle(np.roll(an_signal, DELAY)[slc]), 'k--', alpha=0.5)
ax.text(t[0], np.pi+0.5, '$\phi_{arhilb}$', color='#0099d8')
ax.text(t[-1], np.pi+0.5, '$\phi$', color='#444444')


fig.savefig('results/viz/all.png', dpi=200)