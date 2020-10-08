import numpy as np
import scipy.signal as sg
import pylab as plt

import statsmodels.api as sm



def transition_matrix(f0, fs):
    T = np.array([[np.cos(2*np.pi*f0/fs), -np.sin(2*np.pi*f0/fs)], [np.sin(2*np.pi*f0/fs), np.cos(2*np.pi*f0/fs)]])
    return T

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def inv_sigmoid(x):
    return np.log(x) - np.log(1-x)

class SingleOscillatorModel:
    def __init__(self, f0, fs, state_cov, obs_cov, a0):
        self.transition = a0*transition_matrix(f0, fs)
        self.design = np.array([1, 0])
        self.state_cov = state_cov
        self.obs_cov = obs_cov
        self.state = np.zeros(2)

    def step(self):
        self.state = self.transition @ self.state + np.random.randn(2) * self.state_cov**0.5
        obs = self.design @ self.state + np.random.randn() * self.obs_cov**0.5
        return self.state, obs

    def batch(self, n_steps):
        states = []
        obss = []
        for k in range(n_steps):
            state, obs  = self.step()
            states.append(state)
            obss.append(obs)
        return states, obss


use_sim = 1
if use_sim:
    FS = 250
    f0 = 10
    a0 = 0.99
    state_cov = 1
    obs_cov = 10

    states, obss = SingleOscillatorModel(f0, FS, state_cov, obs_cov, a0).batch(20000)
    # plt.plot()
else:
    data = np.load('/home/kolai/Projects/cfir/data/eegbci_ica/eegbci2_ICA_real_feet_fist.npz')
    FS = data['fs']
    obss = data['eeg'][:20000]
    obss = sg.filtfilt(*sg.butter(1, [3/FS*2, 20/FS*2], 'bandpass'), obss)

fig, ax = plt.subplots(2)
ax[0].plot(obss)
ax[1].plot(*sg.welch(obss, FS), label='Input')




# Construct the model
class AR2(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        # Initialize the state space model
        super(AR2, self).__init__(endog, k_states=2, k_posdef=2,
                                  initialization='stationary')

        # Setup the fixed components of the state space representation
        self['design'] = [1, 0]
        self['transition'] = [[1, 0],
                                  [0, 1]]
        self['selection'] = np.array([[1., 0.], [0., 1.]])


    # Describe how parameters enter the model
    def update(self, params, *args, **kwargs):
        params = super(AR2, self).update(params, *args, **kwargs)
        a0, f0, state_cov, obs_cov = params
        self['transition'] = transition_matrix(f0, FS)*a0
        self['state_cov', 0, 0] = state_cov
        self['state_cov', 1, 1] = state_cov
        self['obs_cov', 0, 0] = obs_cov

    def transform_params(self, unconstrained):
        x = np.hstack((sigmoid(unconstrained[0]), unconstrained[1:]**2))
        return x

    def untransform_params(self, constrained):
        x = np.hstack((inv_sigmoid(constrained[0]), constrained[1:]**0.5))
        return x

    # Specify start parameters and parameter names
    @property
    def start_params(self):
        return [0.9, 1, 0.1, 1]  # these are very simple

    @property
    def param_names(self):
        return ['a0', 'f0', 'state_cov', 'obs_cov']

mod = AR2(obss)
res = mod.fit()
print(res.summary())
sim = mod.simulate(res.params, 20000)
ax[0].plot(sim, '--')
ax[1].plot(*sg.welch(sim, FS), '--', label='Sim. after MLE')
sim = mod.simulate(mod.start_params, 20000)
ylim = ax[1].get_ylim()
ax[1].plot(*sg.welch(sim, FS), ':', label='Sim. before MLE')
ax[1].set_ylim(ylim)
plt.legend()
ax[0].set_xlim(0, FS*10)

ax[0].set_xlabel('Samples')
ax[1].set_xlabel('Freq, Hz')
ax[0].set_title('Simulated' if use_sim else 'Filtered EEG')
plt.tight_layout()
