import h5py
import numpy as np
import pylab as plt
import scipy.signal as sg
import pandas as pd

with h5py.File('/home/kolai/Data/alpha_delay2/alpha2-delay-subj-3_11-07_18-52-37/experiment_data.h5') as f:
    p4_index = list(map(bytes.decode, f['channels'][:])).index('P4')
    x = f['protocol2/raw_data'][:][:, p4_index]
    fs = int(f['fs'].value)
    print(fs)




n_samples = fs*20
x = x[20*fs:]
x = (x - x.mean())/x.std()
freq, pxx = sg.welch(x, fs, nperseg=n_samples, scaling='spectrum')
band = [10.5-1.7, 10.5+1.7]
pxx[(freq>band[0])&(freq <band[1])] = np.nan
pxx = pd.Series(pxx).fillna(0).values
plt.plot(freq, pxx, '.-')
plt.show()



amplitudes = pxx[1:]**0.5
amplitudes = np.concatenate([[0], amplitudes, amplitudes[::-1]])

# random dft coefficients phases
rand_phi = 1j*np.random.uniform(0, np.pi*2, n_samples//2)
exponents = np.exp(np.concatenate([[0], rand_phi, -rand_phi[::-1]]))

# inverse dft
x_sim = np.fft.ifft(amplitudes*exponents)[:n_samples].real
x_sim = (x_sim - x_sim.mean())/x_sim.std()
t = np.arange(n_samples) / fs
amp = sg.filtfilt(*sg.butter(4, (band[1] - band[0]) / fs * 2 * 0.45, 'low'), np.random.normal(size=n_samples*2))[:n_samples]* 15.5
alpha = np.sin(2 * np.pi * t * sum(band) / 2) * amp
x_sim += alpha
x_sim = (x_sim - x_sim.mean())/x_sim.std()
amp = np.abs(amp) /x_sim.std()
alpha = (alpha- x_sim.mean())/x_sim.std()

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
t = np.arange(n_samples)/fs
ax2.plot(t, x_sim, label='sim-eeg')
ax3.plot(t, alpha, label='sim-alpha')
ax3.plot(t, amp, label='sim-alpha-envelope')
ax1.plot(t, x[fs*9:n_samples+fs*9], label='real')
[ax.legend() for ax in (ax1, ax2, ax3)]
plt.xlabel('Time, s')
plt.show()

plt.semilogy(*sg.welch(x, fs, nperseg=500, scaling='spectrum'))
plt.semilogy(*sg.welch(alpha, fs, nperseg=500, scaling='spectrum'))
plt.semilogy(*sg.welch(x_sim, fs, nperseg=500, scaling='spectrum'))
plt.legend(['real', 'sim', 'alpha'])
plt.xlabel('Freq, Hz')
plt.show()