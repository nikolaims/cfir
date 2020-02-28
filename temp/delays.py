import sys
sys.path.insert(0, r'/home/kolai/Projects/nfblab/nfb')
import numpy as np
from utils.load_results import load_data
from release.utils import band_hilbert
import pylab as plt
import h5py
import seaborn as sns


nor = lambda x: (x - x.mean()) / x.std()

with h5py.File(r'/media/kolai/UBUNTU 18_0/delay0-testP4_02-04_19-32-39/experiment_data.h5') as f:
    a = f['protocol3/signals_stats/Alpha0/bandpass'][:]

df, fs, channels, p_names = load_data(r'/media/kolai/UBUNTU 18_0/delay0-testP4_02-04_19-32-39/experiment_data.h5')

# df['log'] = np.abs(band_hilbert(df['P4'].values, fs, a))**0.5
# sns.boxplot('block_number', 'log', data=df.query('block_name=="FB0" | block_name=="Baseline0" | block_name=="Baseline"'))
df = df.query('block_number==5')

start = 1000
fb = df['P4'].values
ph = nor(df['PHOTO'].values[start:])
fb_filt = nor(np.abs(band_hilbert(fb, fs, a))[start:])

# fb_filt = nor(df['signal_Alpha0'].values[start:])

plt.plot(fb_filt)
plt.plot(ph)

delays = np.arange(0, 1000, 10)
corrs = [np.corrcoef(ph, np.roll(fb_filt, d))[1, 0] for d in delays]
opt = delays[np.argmax(corrs)]
print(opt/fs)

plt.plot(corrs)

