import h5py
import numpy as np
import pylab as plt
import scipy.signal as sg
import pandas as pd
import os



data_folder = '/home/kolai/Data/alpha_delay2'
pxxs = []
alpha_hill = []
for exp_folder in map(lambda x: os.path.join(data_folder, x), os.listdir(data_folder)):

    with h5py.File(os.path.join(exp_folder, 'experiment_data.h5')) as f:
        if 'protocol4' not in f or len(f) < 20: continue
        fs = int(f['fs'].value)
        p4_index = list(map(bytes.decode, f['channels'][:])).index('P4')
        x = f['protocol4/raw_data'][:][:, p4_index]

        if len(x)<fs*70 or f['protocol4'].attrs['name']!='Baseline': continue
        print(f['protocol4'].attrs['name'], len(x) / fs, fs)


    n_samples = fs*1
    x = x[30*fs:]
    x = (x - x.mean())/x.std()
    freq, pxx = sg.welch(x, fs, window=np.ones(n_samples), scaling='spectrum')
    alpha_mask = np.logical_and(freq>8, freq<13)
    main_freq = freq[alpha_mask][np.argmax(pxx[alpha_mask])]
    alpha_mask = np.abs(freq - main_freq) <= 2
    alpha_hill.append(pxx[alpha_mask]-np.interp(freq[alpha_mask], freq[~alpha_mask], pxx[~alpha_mask]))
    ppx_no_alpha = np.interp(freq, freq[~alpha_mask], pxx[~alpha_mask])

    plt.plot(main_freq, pxx[freq==main_freq], '.k')
    plt.plot(freq, pxx, 'b', alpha=0.1)
    pxxs.append(ppx_no_alpha)




background_spec = np.exp(np.median(np.log(pxxs), 0))**0.5
background_spec[0] = 0
plt.plot(freq, background_spec, 'r')
plt.xlabel('Freq, Hz')
plt.semilogy()
plt.show()

plt.figure()
plt.plot(freq[alpha_mask], np.clip(alpha_hill, 1e-10, np.inf).T, 'b', alpha=0.1)
plt.plot(freq[alpha_mask], np.exp(np.median(np.log(np.clip(alpha_hill, 1e-10, np.inf)), 0))*3, 'r')

plt.show()



n_samples = fs*100 + 1
freq_full = np.fft.fftfreq(n_samples, 1/fs)
amplitudes = np.interp(np.abs(freq_full), freq, background_spec)

# random dft coefficients phases
rand_phi = 1j*np.random.uniform(0, np.pi*2, n_samples//2)
exponents = np.exp(np.concatenate([[0], rand_phi,  -rand_phi[::-1]]))

# inverse dft
x_sim = np.real_if_close(np.fft.ifft(amplitudes*exponents)[:n_samples])
band = [8, 12]
t = np.arange(n_samples) / fs
amp = sg.filtfilt(*sg.butter(6, (band[1] - band[0]) / fs/2, 'low'), np.random.normal(size=n_samples*2))[:n_samples]
alpha = np.sin(2 * np.pi * t * sum(band) / 2) * amp
x_sim += alpha/500
x_sim = (x_sim - x_sim.mean())/x_sim.std()
#plt.plot(x_sim)
#plt.show()
#
freq_full, pxx_full = sg.welch(x_sim[:fs*120], fs, window=np.ones(fs), scaling='spectrum')

plt.plot(freq_full, pxx_full)
plt.plot(freq, (background_spec**2))
plt.plot(freq, pxx)
plt.semilogy()
plt.show()

