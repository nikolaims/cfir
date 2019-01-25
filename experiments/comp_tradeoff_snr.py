import numpy as np
from pycfir.filters import get_x_chirp, RectEnvDetector, CFIRBandEnvelopeDetector, HilbertWindowFilter, rt_emulate, FiltFiltRectSWFilter
import pylab as plt
import pickle
import pandas as pd
import seaborn as sns


def delay_align(x, y, delay):
    if delay >= 0:
        x = x[delay:]
        y = y[:-delay or None]
    else:
        x = x[:delay]
        y = y[abs(delay):]
    return x, y


def corr_delay(x, y, delay):
    x, y = delay_align(x, y, delay)
    corr = np.corrcoef(x, y)[0, 1]
    return corr

# load sim data
with open('alpha_sim_snr.pkl', 'rb') as handle:
    sim_dict = pickle.load(handle)

# params
n_seconds = 20
fs = 500
delays = np.arange(-50, 100, 5)
snrs = [0, 1, 5]
stats = pd.DataFrame(columns=['method', 'delay', 'corr', 'snr'])

# iterate cross snr-s
for snr in snrs:
    print(snr)
    x = sim_dict['alpha'][:fs*n_seconds] * snr + sim_dict['noise'][:fs * n_seconds]
    amp = sim_dict['envelope'][:fs*n_seconds]

    for j, delay in enumerate(delays):
        # Rect. env. detector
        if delay > 0:
            corrs = []
            for k in range(1, delay):
                method = RectEnvDetector([8, 12], fs, k * 2, 2 * delay - 2 * k)
                corr = corr_delay(method.apply(x), amp, delay)
                corrs.append(0 if np.isnan(corr) else corr)
            opt_corr = np.max(corrs)
        else:
            opt_corr = 0
        stats = stats.append({'method': 'Rect.', 'delay': delay, 'corr': opt_corr, 'snr': snr}, ignore_index=True)

        # cFIR env detector
        method = CFIRBandEnvelopeDetector([8, 12], fs, delay, n_taps=1000, n_fft=2000)
        opt_corr = corr_delay(method.apply(x), amp, delay)
        stats = stats.append({'method': 'cFIR', 'delay': delay, 'corr': opt_corr, 'snr': snr}, ignore_index=True)

        # Hilbert
        if delay >= 0:
            method = HilbertWindowFilter(500, fs, [8, 12], delay)
            opt_corr = corr_delay(np.abs(rt_emulate(method, x, 10)), amp, delay)
        else:
            opt_corr = 0
        stats = stats.append({'method': 'wHilbert', 'delay': delay, 'corr': opt_corr, 'snr': snr}, ignore_index=True)



# viz methods
methods_linestyles = ['-.', '-', '--']
methods = ['Rect.', 'cFIR', 'wHilbert']
#colors = ["windows blue", "amber", "greyish", "faded green", "dusty purple"]
snrs_colors = sns.xkcd_palette(["red", 'amber', 'medium green', 'blue'])

for method, method_linestyle in zip(methods, methods_linestyles):
    plt.plot(np.nan, color='k', linestyle=method_linestyle, label=method)
for snr, snr_color in zip(snrs, snrs_colors):
    plt.plot(np.nan,  color=snr_color, label='SNR={}'.format(snr))
plt.legend()


for method, method_linestyle in zip(methods, methods_linestyles):
    for snr, snr_color in zip(snrs, snrs_colors):
        data = stats.query('method=="{}" & snr=={}'.format(method, snr))
        plt.plot(data['delay'].values/fs*1000, data['corr'].values, color=snr_color, linestyle=method_linestyle, label=None)

plt.axvline(0, color='k', alpha=0.3)
plt.xlabel('Delay, ms')
plt.ylabel('Corr.')
plt.show()