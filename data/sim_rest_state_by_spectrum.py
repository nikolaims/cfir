import numpy as np
import pandas as pd
import h5py
import pylab as plt
import scipy.signal as sg
import os
from seaborn import color_palette
import pickle
from pycfir.utils import interval_mask, individual_band_snr, magnitude_spectrum, interval_flankers_mask
from data.settings import FLANKER_WIDTH, FS, GFP_THRESHOLD, ALPHA_BAND_EXT, ALPHA_BAND_HALFWIDTH, WELCH_NPERSEG, ALPHA_BAND




# collect data
eeg_df = pd.read_pickle('data/rest_state_probes_real.pkl')

# estimate spectra
freq = np.fft.rfftfreq(WELCH_NPERSEG, 1 / FS)
alpha_mask = np.logical_and(freq > ALPHA_BAND_EXT[0], freq < ALPHA_BAND_EXT[1])
background_spectra = []
alpha_peaks = []
for dataset, data in eeg_df.groupby('dataset'):

    x = data['eeg'].values
    # welch boxcar window
    _freq, pxx = magnitude_spectrum(x, FS)
    band, snr = individual_band_snr(x, FS, ALPHA_BAND_EXT, ALPHA_BAND_HALFWIDTH, FLANKER_WIDTH)
    #pxx = sg.welch(x, FS, window=np.ones(WELCH_NPERSEG), scaling='spectrum')[1] ** 0.5

    # find individual alpha mask range

    ind_alpha_mask = interval_mask(freq, band)

    # store alpha peak minus flanker interpolation
    alpha_pxx = pxx[ind_alpha_mask] - np.interp(freq[ind_alpha_mask], freq[~ind_alpha_mask], pxx[~ind_alpha_mask])
    alpha_peaks.append(alpha_pxx)

    # alpha range flanker interpolation
    ppx_no_alpha = np.interp(freq, freq[~ind_alpha_mask], pxx[~ind_alpha_mask])
    background_spectra.append(ppx_no_alpha)

    # viz spectra
    plt.plot(np.mean(band), pxx[freq == np.mean(band)], '.k')
    plt.plot(freq, pxx, 'b', alpha=0.1)

# background spectrum
background_spectrum = np.median(background_spectra, 0)
background_spectrum[0] = 0

# alpha spectrum
alpha_spectrum = np.zeros_like(background_spectrum)
alpha_mask = interval_mask(freq, ALPHA_BAND)
alpha_flankers_mask = interval_flankers_mask(freq, ALPHA_BAND, FLANKER_WIDTH)
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
_, pxx_full = magnitude_spectrum(background_sim, FS)
background_sim *= background_spectrum[alpha_flankers_mask].mean() / pxx_full[alpha_flankers_mask].mean()

# simulate alpha sim
alpha_sim = sim_from_spec(n_seconds_to_sim, freq, alpha_spectrum)

# normalize alpha sim
_, pxx_full = magnitude_spectrum(alpha_sim, FS)
alpha_sim *= alpha_spectrum[alpha_mask].mean()/pxx_full[alpha_mask].mean()

# extract envelope
alpha_sim_an = sg.hilbert(alpha_sim)

# pickle dump
# np.save('data/rest_state_alpha_sim_analytic.npy', alpha_sim_an)
# np.save('data/rest_state_background_sim.npy', background_sim)

# crop edges
crop_samples = 10*FS
background_sim = background_sim[crop_samples:-crop_samples]
alpha_sim = alpha_sim[crop_samples:-crop_samples]
alpha_sim_an = alpha_sim_an[crop_samples:-crop_samples]

# snr normalization
noise_magnitude = magnitude_spectrum(background_sim, FS)[1][alpha_flankers_mask].mean()
noise_magnitude_alpha = magnitude_spectrum(background_sim, FS)[1][alpha_mask].mean()
alpha_magnitude = magnitude_spectrum(alpha_sim, FS)[1][alpha_mask].mean()
alpha_sim *= (4*noise_magnitude**2 - noise_magnitude_alpha**2)**0.5 / alpha_magnitude
alpha_sim_an *= (4*noise_magnitude**2 - noise_magnitude_alpha**2)**0.5 / alpha_magnitude

# viz snrs and specra
snrs = np.linspace(0.1, 2, len(eeg_df['dataset'].unique()))
cm = color_palette('Reds_r', len(snrs))

for color, snr in zip(cm, snrs):
    _freq, sim_spectrum = magnitude_spectrum(background_sim + alpha_sim*snr, FS)

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
fig, axes = plt.subplots(len(snrs), sharex=True, sharey=True)
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

# save sim
for j, snr in enumerate(snrs):
    eeg = background_sim + alpha_sim * snr
    an = alpha_sim_an * snr


    # find individual alpha
    band, snr_ = individual_band_snr(eeg, FS, ALPHA_BAND_EXT, ALPHA_BAND_HALFWIDTH, FLANKER_WIDTH)
    print(snr, snr_, band)

    eeg_df = eeg_df.append(pd.DataFrame({'sim': 1, 'dataset': 'sim{}'.format(j), 'snr': snr,
                           'band_left': ALPHA_BAND[0], 'band_right': ALPHA_BAND[1], 'eeg': eeg, 'an_signal': an}),
                        ignore_index=True)

# save data
eeg_df.to_pickle('data/rest_state_probes.pkl')
