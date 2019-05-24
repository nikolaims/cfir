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



class WHilbertFilter:
    def __init__(self, band, fs, delay, n_taps, n_fft, **kwargs):
        """
        Window bandpass Hilbert transform
        :param band: band of interest
        :param fs: sampling frequency
        :param delay: desired delay. If delay < 0 return nans
        :param n_taps: length of buffer window
        """
        self.fs = fs
        self.band = band
        self.delay = delay
        if self.delay < 0:
            warnings.warn('WHilbertFilter insufficient delay: delay < 0. Filter will return nans')
            self.b = np.ones(n_taps) * np.nan
        else:
            w = np.arange(n_fft)
            F = np.array([np.exp(-2j * np.pi / n_fft * k * np.arange(n_taps)) for k in np.arange(n_fft)])
            F[(w/n_fft*fs < band[0]) | (w/n_fft*fs > band[1])] = 0
            f = np.exp(2j * np.pi / n_fft * (n_taps-delay) * np.arange(n_fft))
            self.b = f.dot(F)[::-1] * 2 / n_fft
        self.a = np.array([1.])
        self.zi = np.zeros(len(self.b) - 1)

    def apply(self, chunk: np.ndarray):
        y, self.zi = sg.lfilter(self.b, self.a, chunk, zi=self.zi)
        return y


class CFIRBandEnvelopeDetector:
    def __init__(self, band, fs, delay, n_taps=500, n_fft=2000, weights=None, **kwargs):
        """
        Complex-valued FIR envelope detector based on analytic signal reconstruction
        :param band: freq. range to apply band-pass filtering
        :param fs: sampling frequency
        :param smoother: smoother class instance to smooth output signal
        :param delay_ms: delay of ideal filter in ms
        :param n_taps: length of FIR
        :param n_fft: length of freq. grid to estimate ideal freq. response
        :weights: least squares weights
        """
        w = np.arange(n_fft)
        H = 2 * np.exp(-2j * np.pi * w / n_fft * delay)
        H[(w / n_fft * fs < band[0]) | (w / n_fft * fs > band[1])] = 0
        F = np.array([np.exp(-2j * np.pi / n_fft * k * np.arange(n_taps)) for k in np.arange(n_fft)])
        if weights is None:
            self.b = F.T.conj().dot(H)/n_fft
        else:
            W = np.diag(weights)
            self.b = np.linalg.solve(F.T.dot(W.dot(F.conj())), (F.T.conj()).dot(W.dot(H)))
        self.a = np.array([1.])
        self.zi = np.zeros(len(self.b)-1)

    def apply(self, chunk: np.ndarray):
        y, self.zi = sg.lfilter(self.b, self.a, chunk, zi=self.zi)
        return y

if __name__ == '__main__':
    import pandas as pd
    dataset = "alpha2-delay-subj-21_12-06_12-15-09"
    eeg_df = pd.read_pickle('data/rest_state_probes.pkl').query('dataset=="{}"'.format(dataset)).iloc[20000:30000]

    x = eeg_df['eeg'].values
    #x = np.random.normal(size=5000)
    band = [8, 12]
    fs = 500
    delay = 200
    weights = np.abs(sg.stft(x, fs, nperseg=2000, nfft=2000, return_onesided=False))[2].mean(1)

    y = np.roll(np.abs(band_hilbert(x, fs, band)), delay)

    rect_filter_y = RectEnvDetector(band, fs, delay, 150).apply(x)
    whilbert_filter_y = np.abs(WHilbertFilter(band, fs, delay, 500, 2000).apply(x))
    cfir_filter_y = np.abs(CFIRBandEnvelopeDetector(band, fs, delay, 500, 2000, weights).apply(x))

    print(np.corrcoef(y, rect_filter_y)[1,0], np.corrcoef(y, whilbert_filter_y)[1,0], np.corrcoef(y, cfir_filter_y)[1,0])
    import pylab as plt
    plt.plot(y)
    plt.plot(rect_filter_y)
    plt.plot(whilbert_filter_y)
    plt.plot(cfir_filter_y)

