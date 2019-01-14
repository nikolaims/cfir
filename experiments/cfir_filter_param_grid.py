import numpy as np
from pycfir.filters import get_x_chirp, CFIRBandEnvelopeDetector, RectEnvDetector
import pylab as plt
import seaborn


fs = 500
x, amp = get_x_chirp(fs)
x += np.random.normal(size=len(x))*0.2
n_tapses = np.array([1000, 2000, 3000])
cm = seaborn.color_palette("nipy_spectral", len(n_tapses))


for kk, n_taps in enumerate(n_tapses):
    delays = np.arange(20, 300, 30)
    opt_corrs = np.zeros(len(delays))
    for j, delay in enumerate(delays):
        print(delay)
        filt = CFIRBandEnvelopeDetector([8, 12], fs, delay, n_taps=1000, n_fft=n_taps)
        y = filt.apply(x)[delay:]
        y_true = amp[:-delay]
        corr = np.corrcoef(y, y_true)[0, 1]
        opt_corrs[j] = corr

    plt.plot(delays, opt_corrs, label=n_taps, color=cm[kk])

plt.legend()





plt.show()