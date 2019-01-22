import numpy as np
import pylab as plt
import scipy.signal as sg


def sim_noise(n_samples, fs, pink_noise_coeff=-.3):
    # 1/f dft coefficients amplitudes
    amplitudes = (np.arange(1, n_samples//2 + 1)/n_samples*2)**pink_noise_coeff
    amplitudes = np.concatenate([[0], amplitudes, amplitudes[::-1]])

    # random dft coefficients phases
    rand_phi = 1j*np.random.uniform(0, np.pi*2, n_samples//2)
    exponents = np.exp(np.concatenate([[0], rand_phi, -rand_phi[::-1]]))

    # inverse dft
    return np.fft.ifft(amplitudes*exponents)[:n_samples].real

def sim_rhytm(n_samples, fs, band):
    np.random.seed(42)
    t = np.arange(n_samples) / fs
    amp = sg.filtfilt(*sg.butter(4, (band[1]-band[0]) / fs * 2, 'low'), np.random.normal(size=n_samples))
    x = np.sin(2*np.pi*t*sum(band)/2) * amp
    return x, np.abs(amp)


fs = 500
n_samples = fs*100

x = sim_noise(n_samples, fs)*30 + sim_rhytm(n_samples, fs, [9, 11])[0]

plt.plot(np.arange(len(x))/n_samples, x)
plt.show()

plt.plot(*sg.welch(x, fs, nperseg=fs))
plt.show()