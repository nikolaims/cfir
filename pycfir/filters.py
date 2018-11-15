import numpy as np
import scipy.signal as sg


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

class SquareSWFilter(SlidingWindowFilter):
    def process_buffer(self):
        return self.buffer[-1]**2



def rt_emulate(wfilter, x, chunk_size=1):
    y = [wfilter.apply(x[k:k+chunk_size]) for k in range(0, len(x), chunk_size)]
    if len(x) % chunk_size:
        y += [wfilter.apply(x[len(x) - len(x)%chunk_size:])]
    return np.concatenate(y)




if __name__ == '__main__':
    import pylab as plt
    fs = 500
    t = np.arange(fs * 60) / fs
    x = sg.chirp(t, f0=8, f1=10, t1=60, method='linear') * sg.filtfilt(*sg.butter(4, 2 / fs * 2, 'low'), np.random.randn(len(t)))
    x= np.arange(len(t))
    #x[:fs] = 0
    y = rt_emulate(SquareSWFilter(500), x, 100)
    plt.plot((x[99::100]**2 - y[99::100]))
    print(np.allclose(x[99::100]**2, y[99::100]))
    plt.show()
