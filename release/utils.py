import numpy as np
import scipy.signal as sg

from release.constants import WELCH_NPERSEG, ALPHA_BAND


def rt_emulate(wfilter, x, chunk_size=1):
    """
    Emulate realtime filter chunks processing
    :param wfilter: filter instance
    :param x: signal to process
    :param chunk_size: length of chunk
    :return: filtered signal
    """
    y = [wfilter.apply(x[k:k+chunk_size]) for k in range(0, len(x), chunk_size)]
    if len(x) % chunk_size:
        y += [wfilter.apply(x[len(x) - len(x)%chunk_size:])]
    return np.concatenate(y)


def band_hilbert(x, fs, band, N=None, axis=-1):
    """
    Non-causal bandpass Hilbert transform to reconstruct analytic narrow-band signal
    :param x: input signal
    :param fs: sampling frequency
    :param band: band of interest
    :param N: fft n parameter
    :param axis: fft axis parameter
    :return: analytic narrow-band signal
    """
    x = np.asarray(x)
    Xf = np.fft.fft(x, N, axis=axis)
    w = np.fft.fftfreq(N or x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = np.fft.ifft(Xf, axis=axis)[:x.shape[0]]
    return 2*x


class SlidingWindowBuffer:
    def __init__(self, n_taps, dtype='float'):
        """
        Sliding window buffer implement wrapper for numpy array to store last n_taps samples of dtype (util class)
        :param n_taps: length of the buffer
        :param dtype: buffer dtype
        """
        self.buffer = np.zeros(n_taps, dtype)
        self.n_taps = n_taps

    def update_buffer(self, chunk):
        if len(chunk) < len(self.buffer):
            self.buffer[:-len(chunk)] = self.buffer[len(chunk):]
            self.buffer[-len(chunk):] = chunk
        else:
            self.buffer = chunk[-len(self.buffer):]
        return self.buffer


def _interval_mask(x, left, right):
    """
    Boolean interval mask
    :return: x \in [left, right]
    """
    return (x >= left) & (x <= right)


def _interval_flankers_mask(x, left, right, flanker_width):
    """
    Boolean flankers mask
    :return: x \in [left-flanker_width, left] and [right, right + flanker_width]
    """
    mask_l = _interval_mask(x, left - flanker_width, left)
    mask_r = _interval_mask(x, right, right + flanker_width)
    return mask_l | mask_r


def magnitude_spectrum(x, fs, nperseg=WELCH_NPERSEG, return_onesided=False):
    """
    Welch magnitude spectrum
    :param x: signal
    :param fs: sampling frequency
    :return: freq, magn_spectrum
    """
    freq, time, pxx = sg.stft(x, fs, nperseg=nperseg, return_onesided=return_onesided, noverlap=int(nperseg*0.9))
    pxx = np.median(np.abs(pxx), 1)
    return freq, pxx


def individual_max_snr_band(x, fs, initial_band=ALPHA_BAND, band_half_width=None, snr_flanker_width=None):
    """
    Specify initial band to individual band by maximizing SNR
    :param x: signal
    :param fs: sampling frequency
    :param initial_band: initial band of search
    :param band_half_width: target band half width, if None set to initial band half width
    :param snr_flanker_width: flankers width to compute SNR, if None set to initial band half width
    :return: band, SNR
    """
    band_half_width = band_half_width or (ALPHA_BAND[1] - ALPHA_BAND[0]) / 2
    snr_flanker_width = snr_flanker_width or (ALPHA_BAND[1] - ALPHA_BAND[0]) / 2
    freq, pxx = magnitude_spectrum(x, fs)
    search_band_mask = _interval_mask(freq, *initial_band)
    best_snr = 0
    best_band = initial_band
    for main_freq in freq[search_band_mask]:
        band = (main_freq - band_half_width, main_freq + band_half_width)
        band_mean_mag = pxx[_interval_mask(freq, *band)].mean()
        flankers_mean_mag = pxx[_interval_flankers_mask(freq, *band, snr_flanker_width)].mean()
        snr = (band_mean_mag - flankers_mean_mag) / flankers_mean_mag
        if snr>best_snr:
            best_snr = snr
            best_band = band
    return best_band, best_snr


if __name__ == "__main__":
    x = np.arange(5)
    print(_interval_mask(x, 1, 2))
    print(_interval_flankers_mask(x, *[1, 2], 1))
    x = np.random.normal(size=100000)
    x = sg.filtfilt(*sg.butter(1, np.array(ALPHA_BAND)/500*2, 'band'), x)
    print(individual_max_snr_band(x, 500))