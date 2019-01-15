import numpy as np
import scipy.signal as sg


def get_x_chirp(fs, f0=10, f1=10):
    np.random.seed(42)
    t = np.arange(fs * 100) / fs
    amp = sg.filtfilt(*sg.butter(4, 2 / fs * 2, 'low'), np.random.randn(len(t)), method='gust')
    x = sg.chirp(t, f0=f0, f1=f1, t1=10, method='linear') * amp
    return x, np.abs(amp)


def rt_emulate(wfilter, x, chunk_size=1):
    y = [wfilter.apply(x[k:k+chunk_size]) for k in range(0, len(x), chunk_size)]
    if len(x) % chunk_size:
        y += [wfilter.apply(x[len(x) - len(x)%chunk_size:])]
    return np.concatenate(y)


def _cLS(X, Y, lambda_=0):
    """
    Complex valued Least Squares with L2 regularisation
    """
    reg = lambda_*np.eye(X.shape[1])
    b = np.linalg.solve(X.T.dot(X.conj()) + reg, (X.T.conj()).dot(Y))
    return b


def _get_ideal_H(n_fft, fs, band, delay=0):
    """
    Estimate ideal delayed analytic filter freq. response
    :param n_fft: length of freq. grid
    :param fs: sampling frequency
    :param band: freq. range to apply band-pass filtering
    :param delay: delay in samples
    :return: freq. response
    """
    w = np.arange(n_fft)
    H = 2*np.exp(-2j*np.pi*w/n_fft*delay)
    H[(w/n_fft*fs<band[0]) | (w/n_fft*fs>band[1])] = 0
    return H


def cfir_win(n_taps, band, fs, delay, n_fft=2000, reg_coeff=0):
    H = _get_ideal_H(n_fft, fs, band, delay)
    F = np.array([np.exp(-2j * np.pi / n_fft * k * np.arange(n_taps)) for k in np.arange(n_fft)])
    return _cLS(F, H, reg_coeff)


def band_hilbert(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    Xf = np.fft.fft(x, N, axis=axis)
    w = np.fft.fftfreq(x.shape[0], d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = np.fft.ifft(Xf, axis=axis)
    return 2*x


class CFIRBandEnvelopeDetector:
    def __init__(self, band, fs, delay=100, n_taps=500, n_fft=2000, reg_coeff=0):
        """
        Complex-valued FIR envelope detector based on analytic signal reconstruction
        :param band: freq. range to apply band-pass filtering
        :param fs: sampling frequency
        :param smoother: smoother class instance to smooth output signal
        :param delay_ms: delay of ideal filter in ms
        :param n_taps: length of FIR
        :param n_fft: length of freq. grid to estimate ideal freq. response
        :param reg_coeff: least squares L2 regularisation coefficient
        """
        H = _get_ideal_H(n_fft, fs, band, delay)
        F = np.array([np.exp(-2j * np.pi / n_fft * k * np.arange(n_taps)) for k in np.arange(n_fft)])
        self.b = _cLS(F, H, reg_coeff)
        self.a = np.array([1.])
        self.zi = np.zeros(len(self.b)-1)

    def apply(self, chunk: np.ndarray):
        y, self.zi = sg.lfilter(self.b, self.a, chunk, zi=self.zi)
        y = np.abs(y)
        return y



class SlidingWindowFilter:
    def __init__(self, n_taps):
        self.buffer = np.zeros(n_taps)

    def apply(self, chunk):
        if len(chunk) < len(self.buffer):
            self.buffer[:-len(chunk)] = self.buffer[len(chunk):]
            self.buffer[-len(chunk):] = chunk
        else:
            self.buffer = chunk[-len(self.buffer):]
        return np.ones(len(chunk))*self.process_buffer()

    def process_buffer(self):
        raise NotImplementedError


class FiltFiltRectSWFilter(SlidingWindowFilter):
    def __init__(self, n_taps, ba_filter, ba_smoother, delay):
        super(FiltFiltRectSWFilter, self).__init__(n_taps)
        self.ba_filter = ba_filter
        self.ba_smoother = ba_smoother
        self.delay = delay

    def process_buffer(self):
        y = sg.filtfilt(*self.ba_filter, self.buffer)
        y = np.abs(y)
        y = sg.filtfilt(*self.ba_smoother, y)
        return y[-self.delay-1]


class RectEnvDetector:
    def __init__(self, band, fs, n_taps_bandpass, n_taps_smooth, smooth_cutoff=None):
        self.b_bandpass = sg.firwin2(n_taps_bandpass, [0, band[0], band[0], band[1], band[1], fs/2], [0, 0, 1, 1, 0, 0], fs=fs)
        if smooth_cutoff is None: smooth_cutoff = band[1]- band[0]
        self.b_smooth = sg.firwin2(n_taps_smooth, [0, smooth_cutoff, smooth_cutoff, fs/2], [1, 1, 0, 0], fs=fs)
        self.zi_bandpass = np.zeros(n_taps_bandpass-1)
        self.zi_smooth = np.zeros(n_taps_smooth-1)

    def apply(self, chunk):
        y, self.zi_bandpass = sg.lfilter(self.b_bandpass, [1.],  chunk, zi=self.zi_bandpass)
        y = np.abs(y)
        y, self.zi_smooth  = sg.lfilter(self.b_smooth, [1.], y, zi=self.zi_smooth)
        return y


class HilbertWindowFilter(SlidingWindowFilter):
    def __init__(self, n_taps, fs, band, delay, pad=False):
        super(HilbertWindowFilter, self).__init__(n_taps)
        self.delay = delay
        self.fs = fs
        self.band= band
        self.pad=pad

    def process_buffer(self):
        x = np.concatenate([self.buffer, self.buffer[::-1]]) if self.pad else self.buffer
        y = band_hilbert(x, self.fs, self.band)
        return y[-self.delay-1]




if __name__== '__main__':
    fs = 500
    x, amp = get_x_chirp(fs)
    x += np.random.normal(size=len(x))*0.2
    delay = 1000
    filt = HilbertWindowFilter(2000, fs, [8, 12], delay)

    import pylab as plt
    plt.plot(np.abs(rt_emulate(filt, x))[delay:])
    plt.plot(amp[:-delay])
    plt.show()