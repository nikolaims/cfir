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
r = 0.999
sigma_phi = 2
sigma_alpha = 20**2
sigma_z = 10

F = np.array([[r, 0], [0, 1]])
q = np.array([0, 2*np.pi*f0/fs])
B = np.eye(2)
Q = np.array([[((1-r)*sigma_alpha)**2, 0], [0, (2*np.pi/fs*sigma_phi)**2]])
R = sigma_z**2

def fx(x, dt):
    return np.dot(F, x) + B.dot(q)

def hx(x):
   # measurement function - convert state into a measurement
   # where measurements are [x_pos, y_pos]
   return np.array([x[0]*np.cos(x[1])])*1.1

points = MerweScaledSigmaPoints(2, alpha=.1, beta=2., kappa=-1)

kf = UnscentedKalmanFilter(dim_x=2, dim_z=1, dt=1, fx=fx, hx=hx, points=points)
kf.x = x_true # initial state
kf.R = np.array([[R]])*1
kf.Q = Q


for t in range(n_steps):

    w = np.array([np.random.randn(), np.random.randn()])
    x_true = fx(x_true, 1) +  (Q**0.5).dot(w)

    z = hx(x_true) + np.random.randn()*R**0.5
    z = eeg[t]

    x_true_list[t] = x_true
    z_list[t] = z
    x_list[t] = kf.x

    kf.predict()
    kf.update(z)

# plt.plot(z_list)
# plt.plot(x_true_list[:, 0]*np.cos(x_true_list[:, 1]))
# plt.plot(*welch(z_list, fs, nfft=fs*4))

# plt.plot(np.abs(x_true_list[:, 0]))
plt.plot(np.abs(x_list[:, 0]), label='UKF {:.3f}'.format(np.corrcoef(envelope[:n_steps], np.abs(x_list[:, 0]))[1, 0]))
plt.plot(envelope[:n_steps], label='non-causal')
plt.plot(cfir_envelope[:n_steps], label='cFIR {:.3f}'.format(np.corrcoef(envelope[:n_steps], cfir_envelope[:n_steps])[1, 0]))
plt.legend()



# plt.plot(x_list[:, 0]*np.cos(x_list[:, 1]))


# plt.plot(eeg[:n_steps])
# plt.plot(z_list)





