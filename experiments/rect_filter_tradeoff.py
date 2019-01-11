import numpy as np
from pycfir.filters import get_x_chirp, RectEnvDetector
import pylab as plt

fs = 500
x, amp = get_x_chirp(fs)
x += np.random.normal(size=len(x))*0.2
n_taps_bandpass_list = np.arange(20, 500, 10)
n_taps_smooth_list = np.arange(20, 500, 10)


delays = np.arange(20, 300, 10)
opt_corrs = np.zeros(len(delays))
for j, delay in enumerate(delays):
    corrs = []
    print(delay)
    for k in range(1, delay):
        n_taps = [k*2, 2*delay-2*k]
        filt = RectEnvDetector([8, 12], fs, *n_taps)
        y = filt.apply(x)[sum(n_taps)//2:]
        y_true = amp[:-sum(n_taps)//2]
        corr = np.corrcoef(y, y_true)[0, 1]
        corrs.append(0 if np.isnan(corr) else corr)
    opt_corrs[j] = np.max(corrs)

plt.plot(delays, opt_corrs)

plt.show()