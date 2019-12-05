import numpy as np
import scipy.signal as sg
import warnings
import padasip as pa
from scipy.linalg import toeplitz

from release.utils import rt_emulate, band_hilbert, SlidingWindowBuffer


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
        :param delay: delay of ideal filter in samples
        :param n_taps: length of FIR
        :param n_fft: length of freq. grid to estimate ideal freq. response
        :weights: least squares weights. If None match WHilbertFilter
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


class AdaptiveCFIRBandEnvelopeDetector:
    def __init__(self, band, fs, delay, n_taps=500, n_fft=2000, weights=None, ada_n_taps=5000, mu=0.8, upd_n_samples=50, **kwargs):
        """
        Time domain adaptive complex-valued FIR envelope detector based on analytic signal reconstruction
        :param band: freq. range to apply band-pass filtering
        :param fs: sampling frequency
        :param delay: delay of ideal filter in ms
        :param n_taps: length of FIR
        :param n_fft: length of freq. grid to estimate ideal freq. response
        :weights: least squares weights. If None match WHilbertFilter
        :param ada_n_taps: length of the window in the past to compute ground trough signal
        :param mu: RLS forgetting factor
        :param kwargs:
        """
        self.rls = pa.filters.FilterRLS(n=n_taps, mu=mu)
        self.rls.w = CFIRBandEnvelopeDetector(band, fs, delay, n_taps, n_fft, weights).b[::-1]
        self.buffer = SlidingWindowBuffer(ada_n_taps + delay)
        self.fs = fs
        self.band = band
        self.ada_n_taps = ada_n_taps
        self.delay = delay
        self.samples_counter = 0
        self.n_taps = n_taps
        self.upd_n_samples = upd_n_samples

    def apply(self, chunk: np.ndarray):
        if len(chunk) <= self.upd_n_samples:
            x = self.buffer.update_buffer(chunk)
            if self.samples_counter>self.buffer.buffer.shape[0]:
                y = band_hilbert(x[:self.ada_n_taps], self.fs, self.band)[self.ada_n_taps // 2]
                self.rls.adapt(y, x[self.ada_n_taps//2+self.delay-self.n_taps + 1:self.ada_n_taps//2 + 1 + self.delay])
            self.samples_counter += len(chunk)
            return sg.lfilter(self.rls.w[::-1], [1.], x)[-len(chunk):]
        else:
            return rt_emulate(self, chunk, self.upd_n_samples)


class ARPredictor:
    def __init__(self, order):
        """
        Auto-regressive predictor based on Yule-Walker AR model parameters estimation
        :param order: AR model order
        """
        self.order = order
        self.ar = None

    def fit(self, x):
        x -= x.mean()
        c = np.correlate(x, x, 'full')
        acov = c[c.shape[0] // 2: c.shape[0] // 2 + self.order + 1]
        ac = acov / acov[0]
        R = toeplitz(ac[:self.order])
        r = ac[1:self.order + 1]
        ar = np.linalg.solve(R, r)
        self.ar = ar[::-1]
        return self

    def predict(self, x, n_steps):
        mean_x = x.mean()
        pred = np.concatenate([x - mean_x, np.zeros(n_steps)])
        for j in range(n_steps):
            pred[x.shape[0] + j] = self.ar.dot(pred[x.shape[0] + j - self.order:x.shape[0] + j])
        return pred + mean_x

    def fit_predict(self, x, n_steps):
        return self.fit(x).predict(x, n_steps)


class FiltFiltARHilbertFilter:
    def __init__(self, band, fs, delay, n_taps, n_taps_edge, ar_order, max_chunk_size=1, butter_order=1, **kwargs):
        """
        Sliding window Hilbert transform with AR prediction of the forward-backward filtered signal
        :param band: freq. range to apply band-pass filtering
        :param fs: sampling frequency
        :param delay: delay of ideal filter in ms
        :param n_taps: length of FIR
        :param n_taps_edge: n_samples to crop and predict
        :param ar_order: AR model order
        :param max_chunk_size: recompute AR model each max_chunk_size samples
        :param butter_order: order of the Butterworh filter
        """
        self.n_taps = n_taps
        self.buffer = SlidingWindowBuffer(self.n_taps)
        self.ba_bandpass = sg.butter(butter_order, [band[0] / fs * 2, band[1] / fs * 2], 'band')
        self.n_taps_edge = n_taps_edge
        self.ar_order = ar_order
        self.band = band
        self.fs = fs
        self.max_chunk_size = max_chunk_size

    def apply(self, chunk):
        if chunk.shape[0] <= self.max_chunk_size:
            x = self.buffer.update_buffer(chunk)
            y = sg.filtfilt(*self.ba_bandpass, x)
            pred = ARPredictor(self.ar_order).fit_predict(y.real[:-self.n_taps_edge], 2*self.n_taps_edge)
            an_signal = sg.hilbert(pred)[-self.n_taps_edge - 1]
            env = np.ones(chunk.shape[0], dtype=complex) * an_signal

            return env

        else:
            return rt_emulate(self, chunk, self.max_chunk_size)


class FiltFiltARHilbertFilterFIR:
    def __init__(self, band, fs, delay, n_taps, n_taps_edge, ar_order, max_chunk_size=1, butter_order=2, **kwargs):
        self.n_taps = n_taps
        self.buffer = SlidingWindowBuffer(self.n_taps)
        self.ba_bandpass = sg.firwin2(64*butter_order, [0, band[0], band[0], band[1], band[1], fs/2], [0, 0, 1, 1, 0, 0], fs=fs), [1., 0]
        self.n_taps_edge = n_taps_edge
        self.ar_order = ar_order
        self.band = band
        self.fs = fs
        self.max_chunk_size = max_chunk_size

    def apply(self, chunk):
        x = np.zeros((len(chunk), self.n_taps+self.n_taps_edge))
        chunk = np.concatenate([np.zeros(self.n_taps-1), chunk, np.zeros(self.n_taps-1)])
        for k in range(len(x)):
            x[k, :self.n_taps] = chunk[k:k+self.n_taps]
        x[:, :self.n_taps] = sg.filtfilt(*self.ba_bandpass, x[:, :self.n_taps], axis=1)
        for k in range(len(x)):
            x[k] = ARPredictor(self.ar_order).fit_predict(x[k, :self.n_taps-self.n_taps_edge], 2*self.n_taps_edge)
        x = sg.hilbert(x, axis=1)[:, -self.n_taps_edge - 1]
        return x


if __name__ == '__main__':
    import pandas as pd
    dataset = "alpha2-delay-subj-21_12-06_12-15-09"
    eeg_df = pd.read_pickle('data/rest_state_probes.pkl').query('dataset=="{}"'.format(dataset))

    x = eeg_df['eeg'].iloc[20000:30000].values
    #x = np.random.normal(size=5000)
    band = [8, 12]
    fs = 500
    delay = 0
    from release.utils import magnitude_spectrum
    _, weights = magnitude_spectrum(eeg_df['eeg'].iloc[10000:20000].values, fs, 2000)
    #weights = np.median(np.abs(sg.stft(eeg_df['eeg'].iloc[10000:20000].values, fs, nperseg=2000, nfft=2000, return_onesided=False))[2], 1)

    y = np.roll(np.abs(band_hilbert(x, fs, band)), delay)

    # rect_filter_y = RectEnvDetector(band, fs, delay, 150).apply(x)
    whilbert_filter_y = np.abs(WHilbertFilter(band, fs, delay, 500, 2000).apply(x))
    cfir_filter_y = np.abs(CFIRBandEnvelopeDetector(band, fs, delay, 500, 2000, weights).apply(x))
    rlscfir_filter_y = np.abs(AdaptiveCFIRBandEnvelopeDetector(band, fs, delay, 500, 2000, weights=None, max_chunk_size=1).apply(x))
    ffiltar_filter_y = np.abs(FiltFiltARHilbertFilterFIR(band, fs, delay, 2000, 25, 50, max_chunk_size=1).apply(x))

    from time import time
    t0 = time()
    # ffiltar_filter_y = np.abs(ARCFIRBandEnvelopeDetector(band, fs, delay, 500, 2000, weights).apply(x))
    print(time()-t0)

    # print(np.corrcoef(y, rect_filter_y)[1,0], np.corrcoef(y, whilbert_filter_y)[1,0], np.corrcoef(y, cfir_filter_y)[1,0])
    # print(np.corrcoef(y, ffiltar_filter_y)[1,0],
    print(np.corrcoef(y, rlscfir_filter_y)[1,0])
    print(np.corrcoef(y, ffiltar_filter_y)[1, 0])
    import pylab as plt
    plt.plot(y)
    # plt.plot(rect_filter_y)
    plt.plot(whilbert_filter_y)
    plt.plot(cfir_filter_y)
    plt.plot(rlscfir_filter_y)
    plt.plot(ffiltar_filter_y)
    # # x = np.sin(10 * 2 * np.pi * np.arange(100) / 500) + np.random.normal(0, 0.1, 100)
    # # # #
    # x1 = band_hilbert(x[:1000], fs, band).real
    # plt.plot(ARPredictor(50).fit_predict(band_hilbert(x[:550], fs, band).real[:500], 500))
    # plt.plot(x1)
    # plt.plot(band_hilbert(x[:550], fs, band).real)


