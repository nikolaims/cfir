
import numpy as np
import pylab as plt
import pandas as pd

import scipy.signal as sg

from filters import CFIRBandEnvelopeDetector, band_hilbert

data = np.load('/home/kolai/Projects/cfir/data/eegbci_ica/eegbci2_ICA_real_feet_fist.npz')

np.random.seed(42)


delay=0

eeg = data['eeg']
fs = data['fs']
labels = data['labels']
eeg = eeg[labels!=2]
labels = labels[labels!=2]
labels = labels==3



Psi = np.cov(eeg[1:], eeg[:-1])[1, 0] / np.var(eeg)
Psi = 0.9
R = np.var(eeg)*(1-Psi**2)




band = np.array([9, 13])
# plt.plot(*sg.welch(eeg, fs))
# [plt.axvline(h) for h in 2*band]

cfir_envelope = 2*np.abs(CFIRBandEnvelopeDetector(band, fs, delay).apply(eeg))
cfir_eeg = 2*CFIRBandEnvelopeDetector(band, fs, delay).apply(eeg).real
f0 = np.mean(band)
n_steps = len(eeg)

x_true_list = np.zeros((n_steps, 2))
z_list = np.zeros(n_steps)
x_list = np.zeros((n_steps, 2))
# a = 0
# phi = 0

x = np.zeros(2)
x_true = np.zeros(2)

P = np.zeros((2, 2))
# sigma = 2
# sigma_z = 2

get_f = lambda f0: np.array([[np.cos(2*np.pi*f0/fs), -np.sin(2*np.pi*f0/fs)], [np.sin(2*np.pi*f0/fs), np.cos(2*np.pi*f0/fs)]])

F = 0.999*get_f(f0)
# Q = np.eye(2) * sigma**2

H = np.array([1, 0])

sigma = 0.1
Q = np.eye(2) * sigma**2
eeg_z = eeg[1]

err = np.random.randn() * R ** 0.5
for t in range(2, n_steps):
    # x_true = F@x_true + np.random.randn()*sigma
    # x_true_list[t] = x_true
    # err = Psi*err + np.random.randn() * R ** 0.5

    ym1 = eeg_z
    eeg_z = eeg[t]
    y = eeg_z

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


def smooth(x, m=fs):
    b = np.ones(m)/m
    return sg.lfilter(b, [1.], x)



kf_envelope = smooth(np.abs(x_list[:, 0] + 1j*x_list[:, 1]))
cfir_envelope = smooth(np.abs(CFIRBandEnvelopeDetector(band, fs, delay).apply(z_list)))





from sklearn.metrics import roc_curve, roc_auc_score
plt.plot(~labels)
plt.plot(kf_envelope)
plt.plot(cfir_envelope)

plt.figure()
plt.plot([0,1], [0, 1])
plt.plot(*roc_curve(~labels, kf_envelope)[:2], label='KF auc = {:.2f}'.format(roc_auc_score(~labels, kf_envelope)))
plt.plot(*roc_curve(~labels, cfir_envelope)[:2], label='cFIR auc = {:.2f}'.format(roc_auc_score(~labels, cfir_envelope)))
plt.legend()


# kf_env = np.abs(x_list[:, 0] + 1j*x_list[:, 1])
# kf_phase = np.angle(x_list[:, 0] + 1j*x_list[:, 1])
# plt.plot(kf_env, label='kf')
# plt.plot(cfir_envelope[:n_steps], label='cFIR')
# plt.legend()
#
# plt.figure()
# plt.plot(z_list + 5)
# plt.plot(x_list[:, 0] - 5)
# plt.legend(['eeg', 'KF'])
#
# plt.plot(*sg.welch(x_list[:, 0], fs))
# plt.plot(*sg.welch(eeg, fs))
# plt.plot(*sg.welch(2*CFIRBandEnvelopeDetector(band, fs, delay).apply(eeg).real, fs))
