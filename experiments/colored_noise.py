
import numpy as np
import pylab as plt
import pandas as pd

import scipy.signal as sg

from filters import CFIRBandEnvelopeDetector

np.random.seed(42)

eeg_df = pd.read_pickle('data/train_test_data.pkl')
dataset = eeg_df.query('snr>0.9')['subj_id'].unique()[0]
eeg_df = eeg_df.query('subj_id=={}'.format(dataset))

delay=0
an_signal = eeg_df['an_signal'].values*1e6
envelope = np.abs(an_signal)
phase = np.angle(an_signal)

eeg = eeg_df['eeg'].values*1e6

eeg = eeg[envelope<np.percentile(envelope, 50)]

Psi = np.cov(eeg[1:], eeg[:-1])[1, 0] / np.var(eeg)
R = np.var(eeg)*(1-Psi**2)

# process = [0]
# process0 = [np.random.randn()]
# for n in range(1, len(eeg)):
#     process.append(np.random.randn()*R**0.5 + Psi * process[-1])
#     process0.append(np.random.randn()*R**0.5)
#
# plt.plot(*sg.welch(process, 500))
# plt.plot(*sg.welch(process0/np.std(process0)*np.std(eeg), 500))
# plt.plot(*sg.welch(eeg, 500))
#
# eeg = eeg_df['eeg'].values*1e6
# plt.plot(eeg)
# plt.plot(process)



fs = 500
eeg = eeg_df['eeg'].values*1e6
band = eeg_df[['band_left', 'band_right']].values[0]
cfir_envelope = 2*np.abs(CFIRBandEnvelopeDetector(band, fs, delay).apply(eeg))
cfir_eeg = 2*CFIRBandEnvelopeDetector(band, fs, delay).apply(eeg).real
f0 = np.mean(band)
n_steps = fs * 50

x_true_list = np.zeros((n_steps, 2))
z_list = np.zeros(n_steps)
x_list = np.zeros((n_steps, 2))
# a = 0
# phi = 0

x = np.zeros(2)

P = np.zeros((2, 2))
# sigma = 2
# sigma_z = 2

get_f = lambda f0: np.array([[np.cos(2*np.pi*f0/fs), -np.sin(2*np.pi*f0/fs)], [np.sin(2*np.pi*f0/fs), np.cos(2*np.pi*f0/fs)]])

F = get_f(f0)
# Q = np.eye(2) * sigma**2

H = np.array([1, 0])

sigma = 1
Q = np.eye(2) * sigma**2

for t in range(2, n_steps):
    ym1 = eeg[t-1]
    y = eeg[t]

    z_list[t] = y

    xm1 = x.copy()
    Pm1 = P.copy()

    x = F.dot(x)
    P = F.dot(P.dot(F.T)) + Q

    z = y - Psi*ym1
    n = z - H.dot(x) + Psi*H.dot(xm1)
    Sigma = H@P@H.T + Psi*H@Pm1@H*Psi + R - H@F@Pm1@H.T*Psi - Psi*H@Pm1@F.T@H.T
    Pxn = P@H.T - F@Pm1@H.T*Psi
    K = Pxn/Sigma
    x = x + K*n
    P = P - (K*Sigma)[:, None] @ K[None, :]

    x_list[t] = x

kf_env = np.abs(x_list[:, 0] + 1j*x_list[:, 1])
kf_phase = np.angle(x_list[:, 0] + 1j*x_list[:, 1])
plt.plot(kf_env, label='KF {:.3f}'.format(np.corrcoef(envelope[:n_steps], kf_env)[1, 0]))
plt.plot(envelope[:n_steps], label='non-causal')
plt.plot(cfir_envelope[:n_steps], label='cFIR {:.3f}'.format(np.corrcoef(envelope[:n_steps], cfir_envelope[:n_steps])[1, 0]))
plt.legend()