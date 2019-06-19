from release.utils import rt_emulate, band_hilbert, SlidingWindowBuffer, magnitude_spectrum
from release.filters import CFIRBandEnvelopeDetector, RectEnvDetector
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
dataset = "alpha2-delay-subj-21_12-06_12-15-09"
eeg_df = pd.read_pickle('data/rest_state_probes_real.pkl').query('dataset=="{}"'.format(dataset))
stats_df = pd.read_pickle('results/stats.pkl').query('dataset=="{}"'.format(dataset))

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
plt.plot(w, np.abs(np.fft.fft(x)), cm['dg'])
plt.plot(w, np.abs(Xf), cm['dg'], alpha=0.8)
setup_gca()
plt.xlim([-20, 20])
plt.savefig(fdir+'ideal_spec.png', dpi=500, transparent=True)
plt.close()

plt.figure(figsize=(4,2))
plt.plot(w, np.abs(np.fft.fft(x)), cm['dg'], alpha=0.1)
plt.plot(w, np.abs(Xf), cm['dg'])
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