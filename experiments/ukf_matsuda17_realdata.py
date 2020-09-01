from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
import numpy as np
import pylab as plt
import pandas as pd
from scipy.signal import welch

from filters import CFIRBandEnvelopeDetector

np.random.seed(42)

eeg_df = pd.read_pickle('data/train_test_data.pkl')
dataset = eeg_df.query('snr>1')['subj_id'].unique()[0]
eeg_df = eeg_df.query('subj_id=={}'.format(dataset))

delay=0
an_signal = eeg_df['an_signal'].values*1e6
envelope = np.abs(an_signal)
phase = np.angle(an_signal)

eeg = eeg_df['eeg'].values*1e6
band = eeg_df[['band_left', 'band_right']].values[0]
cfir_envelope = np.abs(CFIRBandEnvelopeDetector(band, 500, delay).apply(eeg))

fs = 500
f0 = np.mean(band)
n_steps = fs * 50

x_true_list = np.zeros((n_steps, 2))
z_list = np.zeros(n_steps)
x_list = np.zeros((n_steps, 2))
# a = 0
# phi = 0
x_true = np.zeros(2)
x = np.zeros(2)

P = np.zeros(2)
r = 1
# sigma = 2
# sigma_z = 2

F = r*np.array([[np.cos(2*np.pi*f0/fs), -np.sin(2*np.pi*f0/fs)], [np.sin(2*np.pi*f0/fs), np.cos(2*np.pi*f0/fs)]])
# Q = np.eye(2) * sigma**2

H = np.array([1., 0])

x_true_an = np.vstack([np.real(an_signal[:n_steps]), np.imag(an_signal[:n_steps])])
pred = F.dot(x_true_an)

sigma = np.std(x_true_an[0, 1:] - pred[0, :-1])
Q = np.eye(2) * sigma**2

sigma_z = np.std(x_true_an[0] - eeg[:n_steps])*0.25
R = sigma_z**2



for t in range(1, n_steps):
    z = eeg[t]


    x_true_list[t] = x_true
    z_list[t] = z

    x = F.dot(x)
    P = F.dot(P.dot(F.T)) + Q

    y = z - H.dot(x)
    S = H.dot(P.dot(H)) + R

    K = P.dot(H) / S
    x = x + K * y
    P = (np.eye(2) - K[:, None].dot(H[None, :])).dot(P)

    x_list[t] = x

# plt.plot(x_true_list[:, 0])
# plt.plot(z_list)
# plt.plot(x_list[:, 0])
#
# plt.plot(np.abs(x_list[:, 0] + 1j*x_list[:, 1]))
# plt.plot(np.abs(x_true_list[:, 0] + 1j*x_true_list[:, 1]))

kf_env = np.abs(x_list[:, 0] + 1j*x_list[:, 1])
plt.plot(kf_env, label='KF {:.3f}'.format(np.corrcoef(envelope[:n_steps], kf_env)[1, 0]))
plt.plot(envelope[:n_steps], label='non-causal')
plt.plot(cfir_envelope[:n_steps], label='cFIR {:.3f}'.format(np.corrcoef(envelope[:n_steps], cfir_envelope[:n_steps])[1, 0]))
plt.legend()

