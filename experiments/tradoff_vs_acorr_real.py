import numpy as np
from pycfir.filters import get_x_chirp, RectEnvDetector, CFIRBandEnvelopeDetector, HilbertWindowFilter, rt_emulate, FiltFiltRectSWFilter
import pylab as plt
import pickle
import pandas as pd
import seaborn as sns
import scipy.signal as sg

FLANKER_WIDTH = 2
ALPHA_BAND_WIDTH = 2

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
with open('alpha_real.pkl', 'rb') as handle:
    sim_dict = pickle.load(handle)

# params
n_seconds = 20
fs = 500
delays = np.arange(-200, 200, 20)
subjects = np.arange(1, 15, 2)
stats = pd.DataFrame(columns=['method', 'delay', 'corr', 'snr', 'acorr_delay', 'subj'])
methods = ['cFIR', 'Rect', 'wHilbert']

for method in methods:

    for subj in subjects:
        print(method, subj)
        x = sim_dict['raw'][subj]
        amp = sim_dict['envelope'][subj]

        # estimate snr
        freq, pxx = sg.welch(x, fs, nperseg=fs * 2)
        alpha_mask = (freq >= 8) & (freq <= 12)
        main_freq = freq[alpha_mask][np.argmax(pxx[alpha_mask])]
        band = (main_freq - ALPHA_BAND_WIDTH, main_freq + ALPHA_BAND_WIDTH)
        sig = pxx[(freq >= band[0]) & (freq <= band[1])].mean()
        noise = pxx[((freq >= band[0] - FLANKER_WIDTH) & (freq <= band[0])) | (
                (freq >= band[1]) & (freq <= band[1] + FLANKER_WIDTH))].mean()
        snr = sig / noise


        for j, delay in enumerate(delays):
            # cFIR env detector
            opt_corr, y = get_corr(delay, method)
            stats = stats.append({'method': method, 'delay': delay, 'corr': opt_corr, 'snr': snr, 'acorr_delay': None, 'subj': subj}, ignore_index=True)

            for translation in delays[delays<=delay]:
                corr = corr_delay(y, amp, translation)
                stats = stats.append({'method': method, 'delay': translation, 'corr': corr, 'snr': snr, 'acorr_delay': delay, 'subj': subj},
                                     ignore_index=True)



#snrs = [0, 0]

fig, axes = plt.subplots(len(methods)+1, len(subjects), sharex=True, sharey=True)

for j_method, method in enumerate(methods):
    for j_subj, subj in enumerate(subjects):
        ax = axes[j_method, j_subj]
        subj_stats = stats.query('subj=={} & method=="{}"'.format(subj, method))

        for delay, color in zip(delays, sns.color_palette('viridis', len(delays))):
            data = subj_stats.query('acorr_delay=={}'.format(delay))
            ax.plot(data['delay'].values/fs*1000, data['corr'].values, color=color, alpha=1, zorder=-delay)
            ax.scatter([delay/fs*1000], data.query('delay=={}'.format(delay))['corr'].values, s=50, color=color, alpha=1, zorder=-delay-1)



        #data = snr_stats.query('acorr_delay!=acorr_delay')
        #ax.plot(data['delay']/fs*1000, data['corr'], '-k', zorder=-100000)


        corrs = [subj_stats.query('delay<={}'.format(delay))['corr'].max() for delay in delays]

        ax.plot(delays/fs*1000, corrs, '-k', zorder=+100000)



        snr=subj_stats['snr'].values[0]
        ax.set_title('{} SNR={:.2}'.format(method, snr))



        axes[-1, j_subj].plot(delays/fs*1000, corrs, label=method)

axes[-1, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
#plt.tight_layout()
[(ax.grid(), ax.axvline(0, color='k', alpha=0.3), ax.set_xlabel('Delay, ms'), ax.set_ylabel('Corr.'), ax.set_xlim(-400, 400), ax.set_ylim(0, 1)) for ax in axes.flatten()]
plt.show()