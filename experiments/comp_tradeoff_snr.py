import numpy as np
from pycfir.filters import get_x_chirp, RectEnvDetector, CFIRBandEnvelopeDetector, HilbertWindowFilter, rt_emulate, FiltFiltRectSWFilter
import pylab as plt

fs = 500

delays = np.arange(20, 300, 10)
noises = np.linspace(0, 1, 10)

acorr1 = []
acorr2 = []
acorr3 = []

for noise in noises:
    print(noise)
    x, amp = get_x_chirp(fs)
    x += np.random.normal(size=len(x))*noise

    # rect filter grid search
    opt_corrs = np.zeros(len(delays))
    for j, delay in enumerate(delays):
        corrs = []
        for k in range(1, delay):
            n_taps = [k*2, 2*delay-2*k]
            filt = RectEnvDetector([8, 12], fs, k*2, 2*delay-2*k)
            y = filt.apply(x)[sum(n_taps)//2:]
            y_true = amp[:-sum(n_taps)//2]
            corr = np.corrcoef(y, y_true)[0, 1]
            corrs.append(0 if np.isnan(corr) else corr)
        opt_corrs[j] = np.max(corrs)

    acorr1.append(np.mean([max(opt_corrs[:k]) for k in range(1, len(opt_corrs))]))
    # cfir filter
    opt_corrs = np.zeros(len(delays))
    for j, delay in enumerate(delays):
        filt = CFIRBandEnvelopeDetector([8, 12], fs, delay, n_taps=1000, n_fft=2000)
        y = filt.apply(x)[delay:]
        y_true = amp[:-delay]
        corr = np.corrcoef(y, y_true)[0, 1]
        opt_corrs[j] = corr
    acorr2.append(np.mean([max(opt_corrs[:k]) for k in range(1, len(opt_corrs))]))

    # windowed hilbert
    opt_corrs = np.zeros(len(delays))
    for j, delay in enumerate(delays):
        filt = HilbertWindowFilter(500, fs, [8, 12], delay)
        y = np.abs(rt_emulate(filt, x)[delay:])
        y_true = amp[:-delay]
        corr = np.corrcoef(y, y_true)[0, 1]
        opt_corrs[j] = corr
    acorr3.append(np.mean([max(opt_corrs[:k]) for k in range(1, len(opt_corrs))]))

plt.plot(noises, acorr1)
plt.plot(noises, acorr2)
plt.plot(noises, acorr3)
plt.show()