import numpy as np
import scipy.signal as sg
import warnings


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
    w = np.fft.fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = np.fft.ifft(Xf, axis=axis)
    return 2*x


class RectEnvDetector:
    def __init__(self, band, fs, delay, n_taps_bandpass, smooth_cutoff=None, **kwargs):
        """
        Envelope  detector  based  on  rectification  of  the  band-filtered  signal
        :param band: band of interest
        :param fs: sampling frequency
        :param n_taps_bandpass: FIR bandpass filter number of taps
        :param delay: desired delay to determine FIR low-pass filter number of taps
        :param smooth_cutoff: smooth filter cutoff frequency (if None equals to band length)
        """
        if n_taps_bandpass > 0:
            freq = [0, band[0], band[0], band[1], band[1], fs/2]
            gain = [0, 0, 1, 1, 0, 0]
            self.b_bandpass = sg.firwin2(n_taps_bandpass, freq, gain, fs=fs)
            self.zi_bandpass = np.zeros(n_taps_bandpass - 1)
        else:
            self.b_bandpass, self.zi_bandpass = np.array([1., 0]), np.zeros(1)

        if smooth_cutoff is None: smooth_cutoff = band[1] - band[0]

        n_taps_smooth = delay * 2 - n_taps_bandpass
        if n_taps_smooth > 0:
            self.b_smooth = sg.firwin2(n_taps_smooth, [0, smooth_cutoff, smooth_cutoff, fs/2], [1, 1, 0, 0], fs=fs)
            self.zi_smooth = np.zeros(n_taps_smooth - 1)
        elif n_taps_smooth == 0:
            self.b_smooth, self.zi_smooth = np.array([1., 0]), np.zeros(1)
        else:
            warnings.warn('RectEnvDetector insufficient parameters: 2*delay < n_taps_bandpass. Filter will return nans')
            self.b_smooth, self.zi_smooth = np.array([np.nan, 0]), np.zeros(1)

    def apply(self, chunk):
        y, self.zi_bandpass = sg.lfilter(self.b_bandpass, [1.],  chunk, zi=self.zi_bandpass)
        y = np.abs(y)
        y, self.zi_smooth  = sg.lfilter(self.b_smooth, [1.], y, zi=self.zi_smooth)
        return y


class SlidingWindowBuffer:
    def __init__(self, n_taps, dtype='float'):
        """
        Sliding window buffer updated by chunks
        :param n_taps: length of the buffer
        :param dtype: buffer dtype
        """
        self.buffer = np.zeros(n_taps, dtype)
        self.n_taps = n_taps

    def update_buffer(self, chunk):
        '''
        Update buffer from chunk. If chunk length >= n_taps write last n_taps samples from chunk to buffer.
        :param chunk:
        :return:
        '''
        if len(chunk) < len(self.buffer):
            self.buffer[:-len(chunk)] = self.buffer[len(chunk):]
            self.buffer[-len(chunk):] = chunk
        else:
            self.buffer = chunk[-len(self.buffer):]
        return self.buffer


class WHilbertFilter:
    def __init__(self, band, fs, delay, n_taps, max_chunk_size=1, **kwargs):
        """
        Window bandpass Hilbert transform
        :param band: band of interest
        :param fs: sampling frequency
        :param delay: desired delay. If delay < 0 return nans
        :param n_taps: length of buffer window
        :param max_chunk_size: chunk length to realtime emulation
        """
        self.delay = delay
        if self.delay < 0:
            warnings.warn('WHilbertFilter insufficient delay: delay < 0. Filter will return nans')
        self.fs = fs
        self.band = band
        self.buffer = SlidingWindowBuffer(n_taps)
        self.max_chunk_size = max_chunk_size

    def apply(self, chunk):
        if self.delay < 0: return chunk*np.nan
        if chunk.shape[0] < self.buffer.n_taps:
            x = self.buffer.update_buffer(chunk)
            y = band_hilbert(x, self.fs, self.band)
            return y[-self.delay-1] * np.ones(chunk.shape[0])
        else:
            return rt_emulate(self, chunk, self.max_chunk_size)


if __name__ == '__main__':
    x = np.random.normal(size=5000)
    band = [8, 12]
    fs = 500
    delay = 50

    y = np.roll(np.abs(band_hilbert(x, fs, band)), delay)

    rect_filter_y = RectEnvDetector(band, fs, delay, 50).apply(x)
    whilbert_filter_y = np.abs(WHilbertFilter(band, fs, delay, 500).apply(x))

    import pylab as plt
    plt.plot(y)
    plt.plot(rect_filter_y)
    plt.plot(whilbert_filter_y)

