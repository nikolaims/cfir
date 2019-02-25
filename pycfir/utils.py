import scipy.signal as sg
import numpy as np
from settings import WELCH_NPERSEG, ALPHA_BAND_EXT, ALPHA_BAND_HALFWIDTH, FLANKER_WIDTH

def interval_mask(x, interval):
    return (x >= interval[0]) & (x <= interval[1])

def interval_flankers_mask(x, interval, flanker_width):
    mask_l = interval_mask(x, [interval[0] - flanker_width, interval[0]])
    mask_r = interval_mask(x, [interval[1], interval[1] + flanker_width])
    return mask_l | mask_r

def magnitude_spectrum(x, fs):
    freq, pxx = sg.welch(x, fs, window=np.ones(WELCH_NPERSEG), scaling='spectrum')
    return freq, pxx**0.5

def individual_band_snr(x, fs, main_freq_search_band=ALPHA_BAND_EXT, band_half_width=ALPHA_BAND_HALFWIDTH,
                        flanker_width=FLANKER_WIDTH):
    freq, pxx = magnitude_spectrum(x, fs)
    search_band_mask = interval_mask(freq, main_freq_search_band)
    main_freq = freq[search_band_mask][np.argmax(pxx[search_band_mask])]
    band = (main_freq - band_half_width, main_freq + band_half_width)
    band_mean_mag = pxx[interval_mask(freq, band)].mean()
    flankers_mean_mag = pxx[interval_flankers_mask(freq, band, flanker_width)].mean()
    snr = (band_mean_mag - flankers_mean_mag)/ flankers_mean_mag
    return band, max(snr, 0.)

