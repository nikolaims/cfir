from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
import pylab as plt
import mne
import numpy as np

event_id = dict(hands=2, feet=3, rest=1)
subject = 2
runs = [5, 9, 13]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs)
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
eegbci.standardize(raw)  # set channel names
montage = make_standard_montage('standard_1005')
raw.set_montage(montage)

# strip channel names of "." characters
raw.rename_channels(lambda x: x.strip('.'))


events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3, T0=1))

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(raw, events, event_id, tmin=-1, tmax=3, baseline=(-1, 0),proj=True, picks=picks,  preload=True)


from mne.preprocessing import ICA
ica = ICA(n_components=15)
ica.fit(raw)
ica.plot_components()

ica.plot_properties(epochs['feet'], picks=[1])

sources = ica.get_sources(raw)
foot_eeg = sources.get_data(['ICA001'])[0]
fs = int(raw.info['sfreq'])
from scipy.signal import welch
events_all = mne.events_from_annotations(raw)[0]
labels = np.zeros(len(foot_eeg))
for event in events_all:
    labels[event[0]:] = event[2]

plt.plot(*welch(foot_eeg[labels==3], fs))
plt.plot(*welch(foot_eeg[labels==1], fs))
plt.semilogy(*welch(foot_eeg[labels==2], fs))

np.savez('data/eegbci_ica/eegbci{}_ICA_real_feet_fist.npz'.format(subject),
         eeg=foot_eeg, labels=labels, fs=fs, label_names={1: 'Rest', 2: 'Feet', 3: 'Legs'}, events=events_all)