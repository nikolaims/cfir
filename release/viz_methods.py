"""
Figure 2. Visualisation of processing steps of Methods for narrow-band signal envelope estimation
Figure 7:  Envelope and phase estimates obtained by cFIR method for different delay values
"""

from release.utils import rt_emulate, band_hilbert, SlidingWindowBuffer, magnitude_spectrum
from release.filters import CFIRBandEnvelopeDetector, RectEnvDetector, WHilbertFilter
import numpy as np
import scipy.signal as sg
import pandas as pd
import pylab as plt
from release.constants import FS, ALPHA_BAND


def setup_gca():
    ax= plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)


cm = {'b':'#0099d8', 'lb': '#84BCDA', 'r':'#FE4A49', 'g':'#A2A79E', 'dg': '#444444'}

fdir = 'results/viz/methods/'
subj_id = 8
eeg_df = pd.read_pickle('data/train_test_data.pkl').query('subj_id=={}'.format(subj_id))
stats_df = pd.read_pickle('results/stats.pkl').query('subj_id=="{}"'.format(subj_id))

# raw signal
eeg = eeg_df['eeg'].values
t0 = 29000
wt = 1000
t1 = 800
slc = slice(t0,t0+wt)

plt.figure(figsize=(4,2))
plt.plot(eeg[slc], cm['dg'])
setup_gca()
plt.savefig(fdir+'raw_eeg.png', dpi=500, transparent=True)
plt.close()


# full fft
band = eeg_df[['band_left', 'band_right']].values[0]
x = eeg.copy()
Xf = np.fft.fft(x)
w = np.fft.fftfreq(x.shape[0], d=1. / FS)
Xf[(w < band[0]) | (w > band[1])] = 0

plt.figure(figsize=(4,2))
#plt.plot(w, np.abs(np.fft.fft(x)), cm['dg'])
plt.fill_between(w, 0,  np.abs(np.fft.fft(x)), color=cm['dg'], linewidth=2)
plt.plot(w, np.abs(Xf)*2, cm['dg'], alpha=0)
setup_gca()
plt.xlim([-20, 20])
plt.savefig(fdir+'ideal_spec.png', dpi=500, transparent=True)
plt.close()

plt.figure(figsize=(4,2))
#plt.plot(w, np.abs(np.fft.fft(x)), cm['dg'], alpha=0.1)
plt.fill_between(w, 0,  np.abs(Xf), color=cm['dg'], linewidth=2)
#plt.plot(w, np.abs(Xf), cm['dg'])
setup_gca()
plt.xlim([-20, 20])
plt.savefig(fdir+'ideal_spec_crop.png', dpi=500, transparent=True)
plt.close()

# an signal
x = 2*np.fft.ifft(Xf)
plt.figure(figsize=(4,2))
plt.plot(x[slc].real, cm['dg'])
plt.plot(x[slc].imag, cm['dg'], linestyle='--', alpha=0.5)
setup_gca()
plt.savefig(fdir+'ideal_an.png', dpi=500, transparent=True)
plt.close()

plt.figure(figsize=(4,2))
plt.plot(np.abs(x[slc]), cm['dg'])
plt.plot(x[slc].real, cm['dg'], linestyle='-', alpha=0.3)
plt.plot(x[slc].imag, cm['dg'], linestyle='--', alpha=0.1)
setup_gca()
plt.savefig(fdir+'ideal_env.png', dpi=500, transparent=True)
plt.close()

plt.figure(figsize=(4,2))
plt.plot(np.angle(x[slc]), cm['dg'])
setup_gca()
plt.savefig(fdir+'ideal_ang.png', dpi=500, transparent=True)
plt.close()

plt.figure(figsize=(13,2))
plt.plot(np.abs(x[t0:t0+wt*3]), cm['dg'])
plt.plot(x[t0:t0+wt*3].real, cm['dg'], linestyle='-', alpha=0.3)
setup_gca()
plt.savefig(fdir+'ideal_env_long.png', dpi=500, transparent=True)
plt.close()


# cfir
delay = 50
params = stats_df.query('method=="{}" & metric=="corr" & delay=="{}"'.format('cfir', delay))['params'].values[0]
filt = CFIRBandEnvelopeDetector(band, FS, delay, params['n_taps'])
y = filt.apply(eeg)

plt.figure(figsize=(4,2))
plt.plot(y[slc].real, cm['b'], linestyle='-', alpha=1)
plt.plot(y[slc].imag, cm['b'], linestyle='--', alpha=0.5)
setup_gca()
plt.savefig(fdir+'cfir_an.png', dpi=500, transparent=True)
plt.close()

plt.figure(figsize=(4,2))
plt.plot(np.abs(y[slc]), cm['b'])
plt.plot(y[slc].real, cm['b'], linestyle='-', alpha=0.3)
plt.plot(y[slc].imag, cm['b'], linestyle='--', alpha=0.1)
setup_gca()
plt.savefig(fdir+'cfir_env.png', dpi=500, transparent=True)
plt.close()

plt.figure(figsize=(4,2))
plt.plot(np.angle(y[slc]), cm['b'])
setup_gca()
plt.savefig(fdir+'cfir_ang.png', dpi=500, transparent=True)
plt.close()


# cfir params
b = np.zeros(wt, dtype=complex)*np.nan
b[t1-500:t1] = CFIRBandEnvelopeDetector(band, FS, delay, 500).b[::-1]
plt.figure(figsize=(4,2))
plt.plot(b.real, cm['r'])
plt.plot(b.imag, cm['r'], linestyle='--', alpha=0.5)

plt.plot(y[slc].real, cm['b'], linestyle='-', alpha=0.)
setup_gca()
plt.savefig(fdir+'cfir_b.png', dpi=500, transparent=True)
plt.close()


#cfir design
b = CFIRBandEnvelopeDetector(band, FS, delay, 500).b
n_fft = 2000
Xf = np.fft.fft(b, n_fft)
w = np.fft.fftfreq(n_fft, d=1. / FS)
k = np.arange(n_fft)
H = 2 * np.exp(-2j * np.pi * k / n_fft * delay)
H[(k / n_fft * FS < band[0]) | (k / n_fft * FS > band[1])] = 0

plt.figure(figsize=(4,2))
plt.plot(k / n_fft * FS, np.abs(H), cm['dg'], linestyle='-', alpha=1)
plt.plot(-k / n_fft * FS, np.abs(H)*0, cm['dg'], linestyle='-', alpha=1)
plt.plot(np.fft.fftshift(w), np.fft.fftshift(np.abs(Xf)), cm['r'], linestyle='-', alpha=1)
plt.xlim([-20, 20])
setup_gca()
plt.savefig(fdir+'cfir_b_design_abs.png', dpi=500, transparent=True)
plt.close()

plt.figure(figsize=(4,2))
plt.plot(k / n_fft * FS, np.angle(H), cm['dg'], linestyle='-', alpha=1)
plt.plot(-k / n_fft * FS, np.angle(H)*0, cm['dg'], linestyle='-', alpha=1)
plt.plot(np.fft.fftshift(w), np.fft.fftshift(np.angle(Xf)), cm['r'], linestyle='-', alpha=1)
plt.xlim([-20, 20])
setup_gca()
plt.savefig(fdir+'cfir_b_design_angle.png', dpi=500, transparent=True)
plt.close()


plt.figure(figsize=(4,2))
we, me =  magnitude_spectrum(eeg, FS)
plt.plot(np.fft.fftshift(we), np.fft.fftshift(me), cm['g'])
plt.xlim([-20, 20])
setup_gca()
plt.savefig(fdir+'raw_spec.png', dpi=500, transparent=True)
plt.close()




#rect


delay = 200
filt = RectEnvDetector(band, FS, delay, 200)
y = sg.lfilter(filt.b_bandpass, [1.], eeg)

plt.figure(figsize=(4,2))
plt.plot(y[slc], cm['b'], linestyle='-', alpha=1)
setup_gca()
plt.savefig(fdir+'rect_1nb.png', dpi=500, transparent=True)
plt.close()

plt.figure(figsize=(4,2))
plt.plot(np.abs(y[slc]), cm['b'], linestyle='-', alpha=1)
plt.plot(y[slc], cm['b'], linestyle='-', alpha=0.3)
setup_gca()
plt.savefig(fdir+'rect_2abs.png', dpi=500, transparent=True)
plt.close()

y2 = sg.lfilter(filt.b_smooth, [1.], np.abs(y))


plt.figure(figsize=(4,2))
plt.plot(y2[slc], cm['b'], linestyle='-', alpha=1)
plt.plot(np.abs(y[slc]), cm['b'], linestyle='-', alpha=0.3)
plt.plot(y[slc], cm['b'], linestyle='-', alpha=0.1)
setup_gca()
plt.savefig(fdir+'rect_3smooth.png', dpi=500, transparent=True)
plt.close()


#rect params
b = np.zeros(wt)*np.nan
b[t1-200:t1] = filt.b_bandpass
plt.figure(figsize=(4,2))
plt.plot(b.real, cm['r'])
plt.plot(y[slc].real, cm['b'], linestyle='-', alpha=0.)
setup_gca()
plt.savefig(fdir+'rect_b_bandpass.png', dpi=500, transparent=True)
plt.close()

b = np.zeros(wt)*np.nan
b[t1-200:t1] = filt.b_smooth
plt.figure(figsize=(4,2))
plt.plot(b.real, cm['r'])
plt.plot(y[slc].real, cm['b'], linestyle='-', alpha=0.)
setup_gca()
plt.savefig(fdir+'rect_b_smooth.png', dpi=500, transparent=True)
plt.close()


# whilbert
delay = 100
filt = WHilbertFilter(band, FS, delay, 200, 2000)
y = filt.apply(eeg)

plt.figure(figsize=(4,2))
plt.plot(y[slc].real, cm['b'], linestyle='-', alpha=1)
plt.plot(y[slc].imag, cm['b'], linestyle='--', alpha=0.5)
setup_gca()
plt.savefig(fdir+'whilb_an.png', dpi=500, transparent=True)
plt.close()

win_slc = slice(550, 750)
win_x = band_hilbert(eeg[slc][win_slc], FS, band, 2000)




plt.figure(figsize=(4,2))
plt.plot(eeg[slc], alpha=0)
#plt.plot(np.arange(wt)[win_slc], win_x.imag, cm['b'], linestyle='--', alpha=0.5)
plt.plot(np.arange(wt)[win_slc], win_x.real, cm['b'])
# plt.plot(np.arange(wt)[win_slc][100], win_x.real[100], 'k', marker='o', markersize=10)
# plt.plot(np.arange(wt)[win_slc][100], win_x.imag[100], 'k', marker='o', markersize=10, alpha=0.5)
setup_gca()
plt.savefig(fdir+'whilb_filt_win.png', dpi=500, transparent=True)
plt.close()

plt.figure(figsize=(4,2))
plt.plot(eeg[slc], alpha=0)
plt.plot(np.arange(wt)[win_slc], win_x.imag, cm['b'], linestyle='--', alpha=0.5)
plt.plot(np.arange(wt)[win_slc], win_x.real, cm['b'])
# plt.plot(np.arange(wt)[win_slc][100], win_x.real[100], 'k', marker='o', markersize=10)
# plt.plot(np.arange(wt)[win_slc][100], win_x.imag[100], 'k', marker='o', markersize=10, alpha=0.5)
setup_gca()
plt.savefig(fdir+'whilb_an_win.png', dpi=500, transparent=True)
plt.close()



plt.figure(figsize=(4,2))
plt.plot(eeg[slc], alpha=0)

plt.plot(np.arange(wt)[win_slc], win_x.real, cm['b'], linestyle='-', alpha=0.3)
plt.plot(np.arange(wt)[win_slc], win_x.imag, cm['b'], linestyle='--', alpha=0.1)
plt.plot(np.arange(wt)[win_slc], np.abs(win_x), cm['b'])
plt.plot(np.arange(wt)[win_slc][100], np.abs(win_x)[100], cm['b'], marker='o', markersize=7, markeredgewidth=2, markerfacecolor='w')
# plt.plot(np.arange(wt)[win_slc][100], win_x.imag[100], 'k', marker='o', markersize=10, alpha=0.5)
setup_gca()
plt.savefig(fdir+'whilb_abs_win.png', dpi=500, transparent=True)
plt.close()


plt.figure(figsize=(4,2))
plt.plot(eeg[slc], alpha=0)
plt.plot(np.arange(wt)[win_slc], np.angle(win_x), cm['b'])
plt.plot(np.arange(wt)[win_slc][100], np.angle(win_x)[100], cm['b'], marker='o', markersize=7, markeredgewidth=2, markerfacecolor='w')
# plt.plot(np.arange(wt)[win_slc][100], win_x.imag[100], 'k', marker='o', markersize=10, alpha=0.5)
setup_gca()
plt.savefig(fdir+'whilb_angle_win.png', dpi=500, transparent=True)
plt.close()


# cfir
delay = 100
filt = CFIRBandEnvelopeDetector(band, FS, delay, 200)
y = filt.apply(eeg)

plt.figure(figsize=(4,2))

plt.plot(y[slc].real, cm['b'], linestyle='-', alpha=0.3)
plt.plot(y[slc].imag, cm['b'], linestyle='--', alpha=0.1)
plt.plot(np.abs(y[slc]), cm['b'], linestyle='-', alpha=1)
plt.plot(win_slc.stop, np.abs(y[slc])[win_slc.stop], cm['b'], linestyle='-', alpha=1, marker='o', markersize=7, markeredgewidth=2, markerfacecolor='w')
setup_gca()
plt.savefig(fdir+'whilb_abs.png', dpi=500, transparent=True)
plt.close()

plt.figure(figsize=(4,2))
plt.plot(np.angle(y[slc]), cm['b'], linestyle='-', alpha=1)

plt.plot(win_slc.stop, np.angle(y[slc])[win_slc.stop], cm['b'], linestyle='-', alpha=1, marker='o', markersize=7, markeredgewidth=2, markerfacecolor='w')
setup_gca()
plt.savefig(fdir+'whilb_phase.png', dpi=500, transparent=True)
plt.close()

# plt.savefig(fdir+'whilb_an_win.png', dpi=500, transparent=True)
# plt.close()



# cfir
fig, axes = plt.subplots(6, 2, figsize=(10,5), sharex='col', sharey='col')
plt.subplots_adjust(hspace=0.1, bottom=0.15, left=0.03, right=0.97)
time = np.arange(slc.stop-slc.start)/FS
for d, delay in enumerate([-50, 0, 50, 100, 150, 200]):
    params = stats_df.query('method=="{}" & metric=="corr" & delay=="{}"'.format('cfir', delay))['params'].values[0]
    filt = CFIRBandEnvelopeDetector(band, FS, delay//2, params['n_taps'])
    y = filt.apply(eeg)
    x = 2*np.fft.ifft(Xf)
    if delay==0: axes[d, 0].plot(time, np.abs(x[slc]), cm['r'], linestyle='-', alpha=0.9)
    if delay!=0: axes[d, 0].plot(time+delay/1000, np.abs(x[slc]), cm['dg'], linestyle='--', alpha=0.8)
    axes[d, 0].plot(time, np.abs(y[slc])/max(np.abs(y[slc]))*max(np.abs(x[slc])), cm['b'], linestyle='-', alpha=1, zorder=-10)
    if delay == 0:axes[d, 1].plot(time, np.angle(x[slc]), cm['r'], linestyle='-', alpha=0.8)
    if delay != 0: axes[d, 1].plot(time+delay/1000, np.angle(x[slc]), cm['dg'], linestyle='--', alpha=0.8)
    axes[d, 1].plot(time, np.angle(y[slc]), cm['b'], linestyle='-', alpha=1, zorder=-10)
    axes[d, 0].set_yticks([])
    axes[d, 0].set_ylabel('{} ms'.format(delay))
    axes[d, 1].set_yticks([-np.pi, 0, np.pi])
    axes[d, 1].set_yticklabels(['$-\pi$', '0', '$\pi$'])
    axes[d, 1].spines['right'].set_visible(False)
    axes[d, 1].spines['top'].set_visible(False)
    axes[d, 0].spines['right'].set_visible(False)
    axes[d, 0].spines['top'].set_visible(False)


axes[0, 1].set_xlim(1.3, 1.8)
axes[0, 1].set_ylim(-4, 4)
axes[0, 0].set_xlim(0, 2)
axes[-1, 0].set_xlabel('Time, s')
axes[-1, 1].set_xlabel('Time, s')

axes[0, 0].set_title('A. Envelope')
axes[0, 1].set_title('B. Phase')


lines = [plt.plot(np.nan,  color=cm['r'], linestyle='-', alpha=0.9)[0],
         plt.plot(np.nan,  color=cm['dg'], linestyle='--', alpha=0.8)[0],
         plt.plot(np.nan,  color=cm['b'], linestyle='-', alpha=1)[0]]
plt.figlegend( lines, ['ideal', 'delayed ideal', 'estimation'], loc = 'lower center', ncol=5, labelspacing=0. )
plt.savefig(fdir+'cfir_mult_d.png', dpi=500)
