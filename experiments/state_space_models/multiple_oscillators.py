import numpy as np
import scipy.signal as sg
import pylab as plt
from scipy.linalg import block_diag
import statsmodels.api as sm



def transition_matrix(f0, fs):
    T = np.array([[np.cos(2*np.pi*f0/fs), -np.sin(2*np.pi*f0/fs)], [np.sin(2*np.pi*f0/fs), np.cos(2*np.pi*f0/fs)]])
    return T

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def inv_sigmoid(x):
    return np.log(x) - np.log(1-x)

class MultipleOscillatorModel:
    def __init__(self, f_list, fs, state_cov_list, obs_cov, a_list):
        self.transition = block_diag(*[a0*transition_matrix(f0, fs) for f0, a0 in zip(f_list, a_list)])
        self.design = np.array([1, 0]*len(f_list))
        self.state_cov = np.repeat(state_cov_list, 2)
        self.obs_cov = obs_cov
        self.state = np.zeros(2*len(f_list))
        self.K = len(f_list)

    def step(self):
        self.state = self.transition @ self.state + np.random.randn(2*self.K) * self.state_cov**0.5
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


use_sim = 0
if use_sim:
    FS = 250
    f0 = [4, 10, 20]
    a0 = [0.95, 0.98, 0.93]
    state_cov = [4, 0.5, 1]
    obs_cov = 0

    states, obss = MultipleOscillatorModel(f0, FS, state_cov, obs_cov, a0).batch(20000)
    plt.plot(*sg.welch(obss, FS))
else:
    data = np.load('/home/kolai/Projects/cfir/data/eegbci_ica/eegbci2_ICA_real_feet_fist.npz')
    FS = data['fs']
    obss = data['eeg'][:20000]
    # obss = sg.filtfilt(*sg.butter(1, [3/FS*2, 20/FS*2], 'bandpass'), obss)

fig, ax = plt.subplots(2)
ax[0].plot(obss)
ax[1].plot(*sg.welch(obss, FS), label='Input')




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

mod = AR2(obss, 3)
res = mod.fit()
print(res.summary())
sim = mod.simulate(res.params, 20000)
ax[0].plot(sim, linewidth=1)
ax[1].plot(*sg.welch(sim, FS), linewidth=1, label='Sim. after MLE')
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


# mod.filter()