import numpy as np
import scipy.signal as sg
import warnings
import padasip as pa

from scipy.linalg import toeplitz


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


class SlidingWindowBuffer:
    def __init__(self, n_taps, dtype='float'):
        self.buffer = np.zeros(n_taps, dtype)
        self.n_taps = n_taps

    def update_buffer(self, chunk):
        if len(chunk) < len(self.buffer):
            self.buffer[:-len(chunk)] = self.buffer[len(chunk):]
            self.buffer[-len(chunk):] = chunk
        else:
            self.buffer = chunk[-len(self.buffer):]
        return self.buffer

class ARCFIRBandEnvelopeDetector:
    def __init__(self, band, fs, delay, n_taps=500, n_fft=2000, weights=None, n_taps_buffer=2000, n_taps_edge=3, ar_order=25,**kwargs):
        self.cfir_filter = CFIRBandEnvelopeDetector(band, fs, delay+n_taps_edge, n_taps, n_fft, weights)
        self.buffer = SlidingWindowBuffer(n_taps_buffer, 'complex')
        self.n_taps_edge = n_taps_edge
        self.ar_order = ar_order
        self.delay = delay

    def apply(self, chunk: np.ndarray):
        y = self.cfir_filter.apply(chunk)
        res = np.zeros_like(y)
        for k in range(chunk.shape[0]):
            x = self.buffer.update_buffer([y[k]])
            pred_real = ARPredictor2(self.ar_order).fit_predict(x.real, self.n_taps_edge)
            pred_imag = ARPredictor2(self.ar_order).fit_predict(x.imag, self.n_taps_edge)
            res[k] = pred_real + 1j*pred_imag
        res[np.isnan(res)] = 0
        return res

class ARPredictor:
    def __init__(self, order):
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

class ARPredictor2:
    def __init__(self, order):
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
        pred = SlidingWindowBuffer(self.order)
        pred.buffer = x[-self.order:] - mean_x
        for j in range(n_steps):
            pred.update_buffer([self.ar.dot(pred.buffer)])
        return pred.buffer[-1] + mean_x

    def fit_predict(self, x, n_steps):
        return self.fit(x).predict(x, n_steps)

class FiltFiltARHilbertFilter:
    def __init__(self, band, fs, delay, n_taps, n_taps_edge, ar_order, max_chunk_size, butter_order=2, **kwargs):
        self.n_taps = n_taps
        self.buffer = SlidingWindowBuffer(self.n_taps)
        self.ba_bandpass = sg.butter(butter_order, [band[0] / fs * 2, band[1] / fs * 2], 'band')
        self.delay = delay
        self.n_taps_edge_left = n_taps_edge
        self.n_taps_edge_right = max(n_taps_edge, n_taps_edge - delay)
        self.ar_order = ar_order
        self.band = band
        self.fs = fs
        self.max_chunk_size = max_chunk_size

    def apply(self, chunk):
        if chunk.shape[0] <= self.max_chunk_size:
            x = self.buffer.update_buffer(chunk)
            y = sg.filtfilt(*self.ba_bandpass, x)
            if self.delay < self.n_taps_edge_left:

                pred = ARPredictor(self.ar_order).fit_predict(y.real[:-self.n_taps_edge_left],
                                                              self.n_taps_edge_left + self.n_taps_edge_right)

                an_signal = sg.hilbert(pred)
                env = an_signal[-self.n_taps_edge_right - self.delay - len(
                    chunk) + 1:-self.n_taps_edge_right - self.delay + 1] * np.ones(chunk.shape[0])

            else:
                env = sg.hilbert(y)[-self.delay - chunk.shape[0] + 1:-self.delay + 1] * np.ones(chunk.shape[0])

            return env

        else:
            return rt_emulate(self, chunk, self.max_chunk_size)