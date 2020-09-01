import pandas as pd
import pylab as plt
import numpy as np
import seaborn as sns
from filters import CFIRBandEnvelopeDetector, RectEnvDetector
from utils import magnitude_spectrum
from constants import FS, DELAY_RANGE
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.linear_model import LinearRegression
nor = lambda x: (x - x.mean())/x.std()


eeg_df = pd.read_pickle('data/train_test_data.pkl')
dataset = eeg_df.query('snr>1')['subj_id'].unique()[0]
eeg_df = eeg_df.query('subj_id=={}'.format(dataset))

delay=0
envelope = eeg_df['an_signal'].abs().values*1e6
eeg = eeg_df['eeg'].values*1e6
band = eeg_df[['band_left', 'band_right']].values[0]
cfir_envelope = np.abs(CFIRBandEnvelopeDetector(band, FS, delay).apply(eeg))

print(np.corrcoef(envelope[:-delay if delay>0 else None], cfir_envelope[delay:])[1, 0])

# plt.plot(envelope)
# plt.plot(cfir_envelope)


# sns.kdeplot(nor(envelope)-nor(cfir_envelope))

linreg = LinearRegression()
linreg.fit(cfir_envelope[:, None], envelope)
# plt.plot(linreg.predict(cfir_envelope[:, None]))
cfir_envelope_nor = linreg.predict(cfir_envelope[:, None])

n_lags = 5
lags = np.array([np.roll(envelope, k) for k in range(1, n_lags + 1)]).T[:, ::-1]
linreg2 = LinearRegression()
linreg2.fit(lags, envelope)
# plt.plot(linreg2.predict(lags))
# plt.plot(envelope)

Q = np.zeros((n_lags, n_lags))
Q[-1, -1] = np.var(linreg2.predict(lags) - envelope)
R = np.var(cfir_envelope_nor - envelope)


P = R * np.eye(n_lags)
F = np.diag([1.]*(n_lags-1), 1)
F[-1, :] = linreg2.coef_
H = np.zeros(n_lags)
H[-1] = 1.
B = H

x = np.zeros(n_lags)
n_steps = 10000

x_hist = [0]
k_hist = []
for k in range(1, n_steps):
    if k < n_lags:
        x = np.roll(x, -1)
        x[-1] = cfir_envelope_nor[k]
        x_hist.append(x[-1])
    else:
        z = cfir_envelope_nor[k]
        x = F.dot(x) + B*linreg2.intercept_
        P = F.dot(P.dot(F.T)) + Q

        y = z - H.dot(x)
        S = H.dot(P.dot(H)) + R

        K = P.dot(H)/S
        k_hist.append(K)
        x = x + K*y
        P = (np.eye(n_lags) - K[:, None].dot(H[None, :])).dot(P)

        x_hist.append(x[-1])

plt.plot(x_hist)
plt.plot(envelope[:n_steps])
plt.plot(cfir_envelope_nor[:n_steps])

print(np.corrcoef(envelope[:n_steps], x_hist[delay:])[1, 0])

# plt.plot(np.array(k_hist)[:, -1])
plt.legend(['KF out', 'Ideal env.', 'cFIR env.'])
plt.xlabel('Samples [500Hz]')