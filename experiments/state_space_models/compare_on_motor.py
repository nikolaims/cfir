from sklearn.metrics import roc_curve, roc_auc_score
from filters import CFIRBandEnvelopeDetector, band_hilbert
from experiments.kalman_filters import ColoredMeasurementNoiseKF, SimpleKF, FixedLagKF
import numpy as np
import scipy.signal as sg
import pylab as plt
from scipy.linalg import block_diag
import statsmodels.api as sm

from sklearn.linear_model.logistic import LogisticRegression

def transition_matrix(f0, fs):
    T = np.array([[np.cos(2*np.pi*f0/fs), -np.sin(2*np.pi*f0/fs)], [np.sin(2*np.pi*f0/fs), np.cos(2*np.pi*f0/fs)]])
    return T

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def inv_sigmoid(x):
    return np.log(x) - np.log(1-x)


# Construct the model
class AR2(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, n_oscillators):
        # Initialize the state space model
        super(AR2, self).__init__(endog, k_states=2*n_oscillators, k_posdef=2*n_oscillators,
                                  initialization='stationary')

        # Setup the fixed components of the state space representation
        self['design'] = [1, 0]*n_oscillators
        self['transition'] = np.eye(2*n_oscillators)
        self['selection'] = np.eye(2*n_oscillators)
        self.n_osc = n_oscillators


    # Describe how parameters enter the model
    def update(self, params, *args, **kwargs):
        params = super(AR2, self).update(params, *args, **kwargs)
        a_list = params[:self.n_osc]
        f_list = params[self.n_osc:2*self.n_osc]
        state_cov_list = params[2*self.n_osc:3*self.n_osc]
        obs_cov = params[-1]
        self['transition'] = block_diag(*[a0*transition_matrix(f0, FS) for f0, a0 in zip(f_list, a_list)])
        for k in range(self.n_osc):
            self['state_cov', 2*k, 2*k] = state_cov_list[k]
            self['state_cov', 2*k+1, 2*k+1] = state_cov_list[k]
        self['obs_cov', 0, 0] = obs_cov

    def transform_params(self, unconstrained):
        x = np.hstack((sigmoid(unconstrained[:self.n_osc]), unconstrained[self.n_osc:]**2))
        return x

    def untransform_params(self, constrained):
        x = np.hstack((inv_sigmoid(constrained[:self.n_osc]), constrained[self.n_osc:]**0.5))
        return x

    # Specify start parameters and parameter names
    @property
    def start_params(self):
        return [0.9]*self.n_osc + list(np.arange(self.n_osc)*9+1) + list(1/(np.arange(self.n_osc)+1)) + [1]  # these are very simple

    @property
    def param_names(self):
        return ['a{}'.format(k) for k in range(self.n_osc)] + ['f{}'.format(k) for k in range(self.n_osc)] + \
               ['state_cov{}'.format(k) for k in range(self.n_osc)] + ['obs_cov']


data = np.load('/home/kolai/Projects/cfir/data/eegbci_ica/eegbci2_ICA_real_feet_fist.npz')
np.random.seed(42)

delay=0

# load data balanced
events = data['events']
eeg = data['eeg']
labels = data['labels']
FS = data['fs']
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


mod = AR2(eeg[:10000], 3)
res = mod.fit()
print(res.summary())

mod2 = AR2(eeg[10000:], 3)
res2 = mod2.filter(res.params)


def smooth(x, m=1):
    b = np.ones(m)/m
    return sg.lfilter(b, [1.], x)


features = []
features2 = []
for k in range(0, 3):
    features.append(smooth(np.abs(res.filtered_state[2*k] + 1j*res.filtered_state[2*k+1])))
    features2.append(smooth(np.abs(res2.filtered_state[2 * k] + 1j * res2.filtered_state[2 * k + 1])))
features = np.array(features).T
features2 = np.array(features2).T
logreg = LogisticRegression()
logreg.fit(features, labels[:10000, None])
logreg_p = logreg.predict_proba(features2)[:, 1]
print(logreg.coef_)


skf_envelope = smooth(np.abs(res2.filtered_state[4] + 1j*res2.filtered_state[5]))


cfir_envelope = smooth(2*np.abs(CFIRBandEnvelopeDetector(res.params[3+1]+np.array([-3, 3]), FS, 0, n_taps=FS*2, n_fft=4*FS).apply(eeg[10000:])))


plt.plot(labels[10000:], label='label')
plt.plot(cfir_envelope, label='cFIR')
plt.plot(skf_envelope, label='KF')

plt.legend()


plt.figure()
plt.plot([0,1], [0, 1], 'k--', alpha=0.5)

plt.plot(*roc_curve(labels[10000:], cfir_envelope)[:2], label='cFIR auc = {:.2f}'.format(roc_auc_score(labels[10000:], cfir_envelope)), color='k')
plt.plot(*roc_curve(labels[10000:], skf_envelope)[:2], label='KF auc = {:.2f}'.format(roc_auc_score(labels[10000:], skf_envelope)))
# plt.plot(*roc_curve(labels[10000:], logreg_p)[:2], label='Logreg auc = {:.2f}'.format(roc_auc_score(labels[10000:], logreg_p)))
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()