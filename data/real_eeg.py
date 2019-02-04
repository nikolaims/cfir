import numpy as np
import h5py
import pylab as plt
import scipy.signal as sg
import os
from seaborn import color_palette
import pickle
from pycfir.filters import band_hilbert

ALPHA_BAND_EXT = (7, 13)
ALPHA_BAND_HALFWIDTH = 2
ALPHA_MAIN_FREQ = 10
FS = 500
WELCH_NPERSEG = FS


def load_raw_p4_rest_probes():
    # load 100 second probes (Baseline) for experiments in data_folder
    data_folder = '/home/kolai/Data/alpha_delay2'
    for exp_folder in map(lambda x: os.path.join(data_folder, x), os.listdir(data_folder)):
        with h5py.File(os.path.join(exp_folder, 'experiment_data.h5')) as f:
            if 'protocol4' not in f or len(f) < 20: continue
            fs_ = int(f['fs'].value)
            p4_index = list(map(bytes.decode, f['channels'][:])).index('P4')
            x = f['protocol4/raw_data'][:][20 * FS:FS * 120, p4_index]
            if len(x)<FS*70 or f['protocol4'].attrs['name']!= 'Baseline' or fs_!=FS: continue
            print(f['protocol4'].attrs['name'], len(x) / FS, FS)
            yield x


# collect data
rests = np.array([x for x in load_raw_p4_rest_probes()])
x = rests[0]
ys = []

for x in rests[:]:

    freq = np.fft.rfftfreq(WELCH_NPERSEG, 1 / FS)
    alpha_mask = np.logical_and(freq > ALPHA_BAND_EXT[0], freq < ALPHA_BAND_EXT[1])


    pxx = sg.welch(x, FS, window=np.ones(WELCH_NPERSEG), scaling='spectrum')[1] ** 0.5

    # find individual alpha mask range
    main_freq = freq[alpha_mask][np.argmax(pxx[alpha_mask])]
    ind_alpha_mask = np.abs(freq - main_freq) <= ALPHA_BAND_HALFWIDTH
    y = band_hilbert(x, FS, [main_freq-ALPHA_BAND_HALFWIDTH, main_freq+ALPHA_BAND_HALFWIDTH])
    plt.plot(y.real)
    plt.plot(np.abs(y))
    plt.show()

    ys.append(y)

ys = np.array(ys)
sim_dict = {'alpha': ys.real, 'envelope': np.abs(ys), 'raw': rests}
with open('alpha_real.pkl', 'wb') as handle:
    pickle.dump(sim_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)