import scipy.signal as sg
import numpy as np


def _interval_mask(x, interval):
    return (x >= interval[0]) & (x <= interval[1])

def _interval_flankers_mask(x, interval, flanker_width):
    mask_l = _interval_mask(x, [interval[0] - flanker_width, interval[0]])
    mask_r = _interval_mask(x, [interval[1], interval[1] + flanker_width])
    return mask_l | mask_r

def magnitude_spectrum(x, fs):
    freq, pxx = sg.welch(x, fs, window=np.ones(4*fs), scaling='spectrum')
    return freq, pxx**0.5

def individual_band_snr(x, fs, main_freq_search_band, band_half_width, flanker_width):
    freq, pxx = magnitude_spectrum(x, fs)
    search_band_mask = _interval_mask(freq, main_freq_search_band)
    main_freq = freq[search_band_mask][np.argmax(pxx[search_band_mask])]
    band = (main_freq - band_half_width, main_freq + band_half_width)
    #band_mean_mag = np.exp(np.log(pxx[_interval_mask(freq, band)]).mean())
    main_freq_mag = pxx[freq==main_freq][0]
    flankers_mean_mag = pxx[_interval_flankers_mask(freq, band, flanker_width)].mean()
    snr = (main_freq_mag - flankers_mean_mag) / flankers_mean_mag
    return band, max(snr, 0.)

