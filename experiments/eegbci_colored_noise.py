import numpy as np
import pylab as plt
import scipy.signal as sg
from sklearn.metrics import roc_curve, roc_auc_score
from filters import CFIRBandEnvelopeDetector, band_hilbert
from experiments.kalman_filters import ColoredMeasurementNoiseKF

data = np.load('/home/kolai/Projects/cfir/data/eegbci_ica/eegbci2_ICA_real_feet_fist.npz')
np.random.seed(42)

delay=0

# load data balanced
events = data['events']
eeg = data['eeg']
labels = data['labels']
fs = data['fs']
selected = []
for ind, (ev1, ev2) in enumerate(zip(events[:-1], events[1:])):
    l1, l2 = ev1[2], ev2[2]
    if l1 == 3:
        selected.append(ind)
        selected.append(ind+1)

eeg_events= []
label_events = []
for k in range(len(events)-1):
    if k in selected:
        eeg_events.append(eeg[events[k, 0]:events[k+1, 0]])
        label_events.append(labels[events[k, 0]:events[k + 1, 0]])

eeg = np.concatenate(eeg_events)
labels = np.concatenate(label_events)
labels = labels==1

# process parameters
band = np.array([9, 13])
f0 = np.mean(band)
get_f = lambda f0: np.array([[np.cos(2*np.pi*f0/fs), -np.sin(2*np.pi*f0/fs)], [np.sin(2*np.pi*f0/fs), np.cos(2*np.pi*f0/fs)]])
F = 0.999*get_f(f0)

Q_sqrt = 0.1
Q = np.eye(2) * Q_sqrt ** 2

# measurements parameters
H = np.array([1, 0])

# Psi = np.cov(eeg[1:], eeg[:-1])[1, 0] / np.var(eeg)
Psi = 0.999
R = np.var(eeg)*(1-Psi**2)

# init KF
x_list = np.zeros((len(eeg), 2))
kf = ColoredMeasurementNoiseKF(2, 1, F, Q, H, Psi, R)
for t, z in enumerate(eeg):
    kf.step(eeg[t])
    x_list[t] = kf.x


def smooth(x, m=fs):
    b = np.ones(m)/m
    return sg.lfilter(b, [1.], x)

kf_envelope = smooth(np.abs(x_list[:, 0] + 1j*x_list[:, 1]))
cfir_envelope = smooth(2*np.abs(CFIRBandEnvelopeDetector(band, fs, delay).apply(eeg)))

plt.plot(labels)
plt.plot(kf_envelope)
plt.plot(cfir_envelope)

plt.figure()
plt.plot([0,1], [0, 1])
plt.plot(*roc_curve(labels, kf_envelope)[:2], label='KF auc = {:.2f}'.format(roc_auc_score(labels, kf_envelope)))
plt.plot(*roc_curve(labels, cfir_envelope)[:2], label='cFIR auc = {:.2f}'.format(roc_auc_score(labels, cfir_envelope)))
plt.legend()
