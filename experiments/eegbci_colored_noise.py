import numpy as np
import pylab as plt
import scipy.signal as sg
from sklearn.metrics import roc_curve, roc_auc_score
from filters import CFIRBandEnvelopeDetector, band_hilbert
from experiments.kalman_filters import ColoredMeasurementNoiseKF, SimpleKF, FixedLagKF

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
ckf_x_list = np.zeros((len(eeg), 2))
skf_x_list = np.zeros((len(eeg), 2))
ckf = ColoredMeasurementNoiseKF(2, 1, F, Q, H, Psi, R)
skf = SimpleKF(2, 1, F, Q, H, R)
flkf = FixedLagKF(ColoredMeasurementNoiseKF(2, 1, F, Q, H, Psi, R), 15)


for t, z in enumerate(eeg):
    ckf.step(eeg[t])
    skf.step(eeg[t])
    flkf.step(eeg[t])
    ckf_x_list[t] = ckf.x
    skf_x_list[t] = skf.x


def smooth(x, m=fs):
    b = np.ones(m)/m
    return sg.lfilter(b, [1.], x)

ckf_envelope = smooth(np.abs(ckf_x_list[:, 0] + 1j * ckf_x_list[:, 1]))
skf_envelope = smooth(np.abs(skf_x_list[:, 0] + 1j * skf_x_list[:, 1]))
flkf_x_list = np.array(flkf.xSmooth)
flkf_envelope = smooth(np.abs(flkf_x_list[:, 0] + 1j * flkf_x_list[:, 1]))

cfir_envelope = np.roll(smooth(2*np.abs(CFIRBandEnvelopeDetector(band, fs, fs//2, n_taps=fs*2, n_fft=4*fs).apply(eeg))), -fs//2)


plt.plot(labels)
plt.plot(ckf_envelope)
plt.plot(cfir_envelope)
plt.plot(skf_envelope)
plt.plot(flkf_envelope)

plt.figure()
plt.plot([0,1], [0, 1])
plt.plot(*roc_curve(labels, ckf_envelope)[:2], label='CKF auc = {:.2f}'.format(roc_auc_score(labels, ckf_envelope)))
plt.plot(*roc_curve(labels, cfir_envelope)[:2], label='cFIR auc = {:.2f}'.format(roc_auc_score(labels, cfir_envelope)))
plt.plot(*roc_curve(labels, skf_envelope)[:2], label='KF auc = {:.2f}'.format(roc_auc_score(labels, skf_envelope)))
plt.plot(*roc_curve(labels, flkf_envelope)[:2], label='FLKF auc = {:.2f}'.format(roc_auc_score(labels, flkf_envelope)))
plt.legend()
