import pandas as pd
import pylab as plt
import scipy.signal as sg
import numpy as np

target_df_path = '/home/kolai/Data/cfir/eeg_probes.pkl.zip'
fs = 500
df = pd.read_pickle(target_df_path, compression='gzip')
x = df.loc[(df.subj==3)&(df.block_name!='Close'), 'P4'].values

#stft
stft_kwargs = dict(nperseg=fs*2, nfft=fs*20, noverlap=int(fs*0.99*2), window=('tukey', 0.25))
f, t, Sxx = sg.stft(x, fs, **stft_kwargs)

# cut freq
f_band = (5, 20)
f_slice = (f>f_band[0]) & (f<f_band[1])
Sxx[~f_slice] = 0

# init axes
ax_time = plt.subplot2grid((3, 4), (0, 0), colspan=3)
ax_freq = plt.subplot2grid((3, 4), (1, 3), rowspan=2)
ax_time_freq = plt.subplot2grid((3, 4), (1, 0), colspan=3, rowspan=2, sharex=ax_time, sharey=ax_freq)
ax_colorbar = plt.subplot2grid((3*3,4*3), (0, 3*3), colspan=3)

# plot time domain
ax_time.plot(np.arange(len(x)) / fs, x, alpha=0.5)
ax_time.plot(*sg.istft(Sxx, fs, **stft_kwargs))
ax_time.legend(['raw', 'filtered in {}-{}Hz'.format(*f_band)])

# plot freq domain
ax_freq.plot(np.abs(Sxx[f_slice]).mean(1), f[f_slice])

# plot time-freq domain
im = ax_time_freq.pcolormesh(t, f[f_slice], np.abs(Sxx[f_slice]), cmap='nipy_spectral')
plt.colorbar(im, cax=ax_colorbar, orientation='horizontal')


# setup axes
ax_time.set_ylabel('x[n]')
ax_freq.set_xlabel('|X[k]|')
ax_time_freq.set_ylabel('Frequency, Hz')
ax_time_freq.set_xlabel('Time, s')
ax_colorbar.set_xlabel('|X[k,n]|')
plt.subplots_adjust(hspace=0, wspace=0)
[ax.grid() for ax in [ax_freq, ax_time, ax_time_freq]]
plt.show()