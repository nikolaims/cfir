import numpy as np
import h5py
import pylab as plt
import scipy.signal as sg
import os
from seaborn import color_palette
import pickle

ALPHA_BAND_EXT = (7, 13)
ALPHA_BAND_HALFWIDTH = 2
ALPHA_MAIN_FREQ = 10
FS = 500
WELCH_NPERSEG = FS


# collect data
rests = np.load('data/rest_state_probes.npy')

# estimate spectra
freq = np.fft.rfftfreq(WELCH_NPERSEG, 1 / FS)
alpha_mask = np.logical_and(freq > ALPHA_BAND_EXT[0], freq < ALPHA_BAND_EXT[1])
background_spectra = []
alpha_peaks = []
for x in rests:
    # welch boxcar window
    pxx = sg.welch(x, FS, window=np.ones(WELCH_NPERSEG), scaling='spectrum')[1] ** 0.5

    # find individual alpha mask range
    main_freq = freq[alpha_mask][np.argmax(pxx[alpha_mask])]
    ind_alpha_mask = np.abs(freq - main_freq) <= ALPHA_BAND_HALFWIDTH

    # store alpha peak minus flanker interpolation
    alpha_pxx = pxx[ind_alpha_mask] - np.interp(freq[ind_alpha_mask], freq[~ind_alpha_mask], pxx[~ind_alpha_mask])
    alpha_peaks.append(alpha_pxx)

    # alpha range flanker interpolation
    ppx_no_alpha = np.interp(freq, freq[~ind_alpha_mask], pxx[~ind_alpha_mask])
    background_spectra.append(ppx_no_alpha)

    # viz spectra
    plt.plot(main_freq, pxx[freq == main_freq], '.k')
    plt.plot(freq, pxx, 'b', alpha=0.1)

# background spectrum
background_spectrum = np.median(background_spectra, 0)
background_spectrum[0] = 0

# alpha spectrum
alpha_spectrum = np.zeros_like(background_spectrum)
alpha_mask = np.abs(freq - ALPHA_MAIN_FREQ) <= ALPHA_BAND_HALFWIDTH
alpha_spectrum[alpha_mask] = np.median(alpha_peaks, 0)

# viz background spectrum
plt.plot(freq, background_spectrum, 'r')
plt.plot(freq[alpha_mask], (background_spectrum+alpha_spectrum)[alpha_mask])
plt.xlabel('Freq, Hz')
plt.semilogy()
plt.ylim(1e-8, 1e-5)
plt.xlim(0, FS/2)
plt.show()


# simulate background eeg
n_seconds_to_sim = 140
def sim_from_spec(n_seconds, freq, spectrum):
    n_samples = FS * n_seconds + 1

    # frequencies
    freq_full = np.fft.fftfreq(n_samples, 1 / FS)

    # dft coefficients amplitudes
    amplitudes = np.interp(np.abs(freq_full), freq, spectrum)

    # random dft coefficients phases
    rand_phi = 1j*np.random.uniform(0, np.pi*2, n_samples//2)
    exponents = np.exp(np.concatenate([[0], rand_phi,  -rand_phi[::-1]]))

    # inverse dft
    x_sim = np.real_if_close(np.fft.ifft(amplitudes*exponents)[:n_samples])
    return x_sim

# simulate background eeg
background_sim = sim_from_spec(n_seconds_to_sim, freq, background_spectrum)

# normalize background eeg sim
_, pxx_full = sg.welch(background_sim, FS, window=np.ones(FS), scaling='spectrum')
background_sim *= (background_spectrum[alpha_mask]).mean() / (pxx_full[alpha_mask] ** 0.5).mean()

# simulate alpha sim
alpha_sim = sim_from_spec(n_seconds_to_sim, freq, alpha_spectrum)

# normalize alpha sim
_, pxx_full = sg.welch(alpha_sim, FS, window=np.ones(FS), scaling='spectrum')
alpha_sim *= (alpha_spectrum[alpha_mask]).mean()/(pxx_full[alpha_mask]**0.5).mean()

# extract envelope
alpha_sim_an = sg.hilbert(alpha_sim)

# pickle dump
np.save('data/rest_state_alpha_sim_analytic.npy', alpha_sim_an)
np.save('data/rest_state_background_sim.npy', background_sim)

# crop edges
crop_samples = 10*FS
background_sim = background_sim[crop_samples:-crop_samples]
alpha_sim = alpha_sim[crop_samples:-crop_samples]
alpha_sim_an = alpha_sim_an[crop_samples:-crop_samples]

# snr normalization
noise_magnitude = np.mean(sg.welch(background_sim, FS, window=np.ones(FS), scaling='spectrum')[1][alpha_mask] ** 0.5)
alpha_magnitude = np.mean(sg.welch(alpha_sim, FS, window=np.ones(FS), scaling='spectrum')[1][alpha_mask] ** 0.5)
alpha_sim *= noise_magnitude/alpha_magnitude
alpha_sim_an *= noise_magnitude / alpha_magnitude

# viz snrs and specra
snrs = np.concatenate([[0], np.logspace(0, 1, 5)])
cm = color_palette('Reds_r', len(snrs))

for color, snr in zip(cm, snrs):
    sim_spectrum = sg.welch(background_sim + alpha_sim  * snr, FS, window=np.ones(FS), scaling='spectrum')[1] ** 0.5

    plt.plot(freq, sim_spectrum, color=color, label='SNR = {:.2f}'.format(snr))

plt.xlabel('Frequency, Hz')
plt.ylabel('Magnitude, V')
plt.xlim(0, 40)
plt.ylim(1e-7, 1e-4)
plt.plot(freq, background_spectrum, 'k', label='Median real\nP4 spectrum', alpha=0.6)
plt.semilogy()
plt.legend()
plt.show()

# viz ts
fig, axes = plt.subplots(len(snrs), sharex=True)
plt.subplots_adjust(hspace=0)
t = np.arange(len(background_sim))/FS
for color, snr, ax in zip(cm, snrs, axes):
    ax.plot(t, background_sim + alpha_sim * snr)
    ax.set_yticks([])
    ax.set_ylabel('SNR = {:.2f}'.format(snr))
axes[-1].plot(t, alpha_sim * snrs[-1] , alpha=0.6)
axes[-1].plot(t, np.abs(alpha_sim_an) * snrs[-1])
plt.xlim(0, 10)
plt.xlabel('Time, s')
plt.show()
