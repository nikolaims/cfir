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

def get_corr(delay, method_name):
    if method_name == 'Rect':
        if delay > 0:
            corrs = []
            ys = []
            for k in range(1, delay):
                method = RectEnvDetector([8, 12], fs, k * 2, 2 * delay - 2 * k)
                ys.append(method.apply(x))
                corr = corr_delay(ys[-1], amp, delay)
                corrs.append(0 if np.isnan(corr) else corr)
            opt_corr = np.max(corrs)
            y = ys[np.argmax(corrs)]
        else:
            opt_corr = 0
            y = np.zeros(len(x))*np.nan

    elif method_name == 'cFIR':
        method = CFIRBandEnvelopeDetector([8, 12], fs, delay, n_taps=1000, n_fft=2000)
        y = method.apply(x)
        opt_corr = corr_delay(y, amp, delay)

    # Hilbert
    elif method_name=='wHilbert':
        if delay >= 0:
            method = HilbertWindowFilter(500, fs, [8, 12], delay)
            y = np.abs(rt_emulate(method, x, 10))
            opt_corr = corr_delay(y, amp, delay)
        else:
            opt_corr = 0
            y = np.zeros(len(x)) * np.nan
    else: raise TypeError
    return opt_corr, y

# load sim data
with open('alpha_sim_snr.pkl', 'rb') as handle:
    sim_dict = pickle.load(handle)

# params
n_seconds = 20
fs = 500
delays = np.arange(-200, 200, 20)
snrs = [0, 0.5, 1, 1.5, 2, 2.5]
stats = pd.DataFrame(columns=['method', 'delay', 'corr', 'snr', 'acorr_delay'])
methods = ['cFIR', 'Rect', 'wHilbert']

for method in methods:

    for snr in snrs:
        print(method, snr)
        x = sim_dict['alpha'][:fs*n_seconds] * snr + sim_dict['noise'][:fs * n_seconds]
        amp = sim_dict['envelope'][:fs*n_seconds]
        #x = sim_dict['noise']
        #amp = sim_dict['envelope']

        for j, delay in enumerate(delays):
            # cFIR env detector
            opt_corr, y = get_corr(delay, method)
            stats = stats.append({'method': method, 'delay': delay, 'corr': opt_corr, 'snr': snr, 'acorr_delay': None}, ignore_index=True)

            for translation in delays[delays<=delay]:
                corr = corr_delay(y, amp, translation)
                stats = stats.append({'method': method, 'delay': translation, 'corr': corr, 'snr': snr, 'acorr_delay': delay},
                                     ignore_index=True)



#snrs = [0, 0]

fig, axes = plt.subplots(len(methods)+1, len(snrs), sharex=True, sharey=True)

for j_method, method in enumerate(methods):
    for j_snr, snr in enumerate(snrs):
        ax = axes[j_method, j_snr]
        snr_stats = stats.query('snr=={} & method=="{}"'.format(snr, method))

        for delay, color in zip(delays, sns.color_palette('viridis', len(delays))):
            data = snr_stats.query('acorr_delay=={}'.format(delay))
            ax.plot(data['delay']/fs*1000, data['corr'], color=color, alpha=1, zorder=-delay)
            ax.scatter([delay/fs*1000], data.query('delay=={}'.format(delay))['corr'].values, s=50, color=color, alpha=1, zorder=-delay-1)



        #data = snr_stats.query('acorr_delay!=acorr_delay')
        #ax.plot(data['delay']/fs*1000, data['corr'], '-k', zorder=-100000)


        corrs = [snr_stats.query('delay<={}'.format(delay))['corr'].max() for delay in delays]

        ax.plot(delays/fs*1000, corrs, '-k', zorder=+100000)




        ax.set_title('{} SNR={}'.format(method, snr))



        axes[-1, j_snr].plot(delays/fs*1000, corrs, label=method)

axes[-1, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
[(ax.grid(), ax.axvline(0, color='k', alpha=0.3), ax.set_xlabel('Delay, ms'), ax.set_ylabel('Corr.')) for ax in axes.flatten()]
plt.show()