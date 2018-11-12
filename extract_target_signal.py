import pandas as pd
import pylab as plt
import numpy as np
import scipy.signal as sg
from scipy import fftpack


def band_hilbert(x, fs, band, N=None, axis=-1):
    x = np.asarray(x)
    N = N or x.shape[0]
    Xf = fftpack.fft(x, N, axis=axis)
    w = fftpack.fftfreq(N, d=1. / fs)
    Xf[(w < band[0]) | (w > band[1])] = 0
    x = fftpack.ifft(Xf, axis=axis)[:x.shape[0]]
    return 2*x

if __name__ == '__main__':
    fs = 500
    n_samples = fs*100

    env = sg.filtfilt(*sg.butter(4, 1/fs*2, 'low'), np.random.normal(size=n_samples))
    x = env*np.sin(2*np.pi*10*np.arange(n_samples)/fs)

    #env = sg.filtfilt(*sg.butter(4, 1/fs*2, 'low'), np.random.normal(size=n_samples))
    #x = env*np.sin(2*np.pi*10*np.arange(n_samples)/fs)

    noise = np.random.normal(size=n_samples)
    y = x + 0.1*noise
    z = band_hilbert(y, fs, (8, 12), N=n_samples)
    plt.plot(x)
    plt.plot(y, alpha=0.1)
    plt.plot(np.real(z))
    plt.plot(np.abs(z))
    plt.show()