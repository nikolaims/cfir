import numpy as np
import scipy.signal as sg
import padasip as pa
from statsmodels.regression.linear_model import yule_walker



def get_x_chirp(fs, f0=10, f1=10, return_phase=False):
    np.random.seed(42)
    t = np.arange(fs * 30) / fs
    amp = np.abs(sg.filtfilt(*sg.butter(4, 2 / fs * 2, 'low'), np.random.randn(len(t)), method='gust'))
    x = sg.chirp(t, f0=f0, f1=f1, t1=10, method='linear')
    if return_phase:
        return x * amp, amp, np.angle(sg.hilbert(x))
    return x * amp, amp


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
    def __init__(self, band, fs, delay=100, n_taps=500, n_fft=2000, reg_coeff=0, **kwargs):
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
        return y


class ARCFIRBandEnvelopeDetector(CFIRBandEnvelopeDetector):
    def __init__(self, band, fs, delay=100, n_taps=500, n_fft=2000, reg_coeff=0, n_taps_pred=500, ar_order=50, delay_threshold=50):
        if delay < delay_threshold:
            self.enable_prediction = True
            cfir_delay = delay_threshold
            self.buffer = SlidingWindowBuffer(n_taps_pred, 'complex')
            self.ar_order = ar_order
            self.pred_delay = delay_threshold
        else:
            self.enable_prediction = False
            cfir_delay = delay
        super(ARCFIRBandEnvelopeDetector, self).__init__(band, fs, cfir_delay, n_taps, n_fft, reg_coeff)


    def apply(self, chunk: np.ndarray):
        y = super(ARCFIRBandEnvelopeDetector, self).apply(chunk)
        if self.enable_prediction:
            y = self.buffer.update_buffer(y)
            ar_real, s = yule_walker(y.real, self.ar_order, 'mle')
            ar_imag, s = yule_walker(y.imag, self.ar_order, 'mle')

            pred = y.tolist()
            for _ in range(self.pred_delay):
                real = ar_real[::-1].dot(np.array(pred).real[-self.ar_order:])
                imag = ar_imag[::-1].dot(np.array(pred).imag[-self.ar_order:])
                pred.append(real + 1j*imag)
            y = np.array(pred)[-len(chunk):]
        return y





class AdaptiveCFIRBandEnvelopeDetector(CFIRBandEnvelopeDetector):
    def __init__(self, band, fs, delay, n_taps=500, n_fft=2000, reg_coeff=0, ada_n_taps=1000, mu=0.9, max_chunk_size=10, **kwargs):
        super(AdaptiveCFIRBandEnvelopeDetector, self).__init__(band, fs, delay, n_taps, n_fft, reg_coeff)
        self.rls = pa.filters.FilterRLS(n=len(self.b), mu=mu)
        self.rls.w = self.b[::-1]
        self.buffer = SlidingWindowBuffer(len(self.b) +  ada_n_taps // 2 - delay)
        self.fs = fs
        self.band = band
        self.ada_n_taps = ada_n_taps
        self.delay = delay
        self.max_chunk_size = max_chunk_size

    def apply(self, chunk: np.ndarray):
        if len(chunk) <= self.max_chunk_size:
            x = self.buffer.update_buffer(chunk)
            y = band_hilbert(x[-self.ada_n_taps:], self.fs, self.band)[-self.ada_n_taps//2-1]
            self.rls.adapt(y, x[:len(self.b)])
            self.rls.w -= 0.0000001 * self.rls.w
            self.b = self.rls.w[::-1]
            y = sg.lfilter(self.b, self.a, x)
            return y[-len(chunk):]
        else:
            return rt_emulate(self, chunk, self.max_chunk_size)

class AdaptiveEnvelopePredictor:
    def __init__(self, env_detector, n_taps, delay):
        self.rls = pa.filters.FilterRLS(n=n_taps, mu=0.9)
        self.buffer = SlidingWindowBuffer(max(n_taps - delay, n_taps))
        self.delay = delay
        self.n_taps = n_taps
        self.env_detector = env_detector

    def apply(self, chunk: np.ndarray):
        env = np.abs(self.env_detector.apply(chunk))
        env = self.buffer.update_buffer(env)
        self.rls.adapt(env[min(-self.delay, -1)], env[:self.n_taps])
        self.rls.w -= 0.0001 * self.rls.w
        y = self.rls.predict(env[-self.n_taps:])
        return y * np.ones(len(chunk))



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


class WHilbertFilter:
    def __init__(self, n_taps, fs, band, delay, max_chunk_size, **kwargs):
        self.delay = delay
        self.fs = fs
        self.band = band
        self.buffer = SlidingWindowBuffer(n_taps)
        self.max_chunk_size = max_chunk_size

    def apply(self, chunk):
        if chunk.shape[0] < self.buffer.n_taps:
            x = self.buffer.update_buffer(chunk)
            y = band_hilbert(x, self.fs, self.band)
            return y[-self.delay-1] * np.ones(chunk.shape[0])
        else:
            return rt_emulate(self, chunk, self.max_chunk_size)



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


class FiltFiltARHilbertFilter:
    def __init__(self, band, fs, n_taps_edge, delay, ar_order, max_chunk_size, butter_order=1, buffer_s=1, **kwargs):
        n_taps_buffer = buffer_s*fs
        self.buffer = SlidingWindowBuffer(n_taps_buffer)
        self.ba_bandpass = sg.butter(butter_order, [band[0]/fs*2, band[1]/fs*2], 'band')
        self.delay = delay
        self.n_taps_edge_left = n_taps_edge
        self.n_taps_edge_right = max(n_taps_edge, n_taps_edge-delay)
        self.ar_order = ar_order
        self.band = band
        self.fs = fs
        self.max_chunk_size=max_chunk_size

    def apply(self, chunk):
        if chunk.shape[0] <= self.max_chunk_size:
            x = self.buffer.update_buffer(chunk)
            y = sg.filtfilt(*self.ba_bandpass, x)
            if self.delay < self.n_taps_edge_left:

                ar, s = yule_walker(y.real[:-self.n_taps_edge_left], self.ar_order, 'mle')

                pred = y.real[:-self.n_taps_edge_left].tolist()
                for _ in range(self.n_taps_edge_left + self.n_taps_edge_right):
                    pred.append(ar[::-1].dot(pred[-self.ar_order:]))

                an_signal = sg.hilbert(pred)
                env = an_signal[-self.n_taps_edge_right-self.delay-len(chunk)+1:-self.n_taps_edge_right-self.delay+1]*np.ones(len(chunk))

                #
                # plt.plot(x, alpha=0.1)
                # plt.plot(pred, alpha=0.9)
                # plt.plot(np.abs(an_signal))
                # plt.plot(y[:-self.n_taps_edge_left], 'k')
                # plt.plot(y, 'k--')
                #
                # plt.show()


            else:
                env = sg.hilbert(y)[-self.delay-len(chunk)+1:-self.delay+1] * np.ones(len(chunk))

            return env

        else:
            return rt_emulate(self, chunk, self.max_chunk_size)


class RectEnvDetector:
    def __init__(self, band, fs, n_taps_bandpass, delay, smooth_cutoff=None, **kwargs):
        if n_taps_bandpass > 0:
            self.b_bandpass = sg.firwin2(n_taps_bandpass, [0,band[0],band[0],band[1],band[1],fs/2], [0,0,1,1,0,0],fs=fs)
            self.zi_bandpass = np.zeros(n_taps_bandpass - 1)
        else:
            self.b_bandpass, self.zi_bandpass = np.array([1., 0]), np.zeros(1)

        n_taps_smooth = delay * 2 - n_taps_bandpass
        if smooth_cutoff is None: smooth_cutoff = band[1]- band[0]
        if n_taps_smooth > 0:
            self.b_smooth = sg.firwin2(n_taps_smooth, [0, smooth_cutoff, smooth_cutoff, fs/2], [1, 1, 0, 0], fs=fs)
            self.zi_smooth = np.zeros(n_taps_smooth - 1)
        else:
            self.b_smooth, self.zi_smooth = np.array([1., 0]), np.zeros(1)

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
    x += np.random.normal(size=len(x))*0.0001  #+ np.sin(2*np.pi*5*np.arange(len(x))/fs)
    delay = 0
    #filt = RectEnvDetector([8, 12], fs, 100, delay)
    #filt = WHilbertFilter(500, fs, [8, 12], delay)
    #filt = CFIRBandEnvelopeDetector([8,12], fs, delay)
    filt = AdaptiveCFIRBandEnvelopeDetector([8, 12], fs, delay, ada_n_taps=500)
    #filt = AdaptiveEnvelopePredictor(filt, 500, -50)
    #filt = FiltFiltARHilbertFilter([8,12], fs, 20, delay, 50, 10, buffer_s=4, butter_order=1)
    #filt = ARCFIRBandEnvelopeDetector([8,12], fs, delay, delay_threshold=50)

    import pylab as plt
    y_hat = np.abs(filt.apply(x)[delay if delay>0 else None:delay if delay<0 else None])
    y = amp[-delay if delay<0 else None:-delay if delay>0 else None]
    plt.plot(y_hat)
    plt.plot(y)
    plt.show()

    print(np.corrcoef(y[1000:], y_hat[1000:])[1,0])

    #plt.figure()
    #plt.plot(filt.rls.w)
    #plt.plot(AdaptiveEnvelopePredictor(filt, 500, 0).rls.w)