import numpy as np
import pylab as plt
import scipy.signal as sg
from scipy.stats import linregress, ttest_ind
import pandas as pd
fs = 500

eeg_df = pd.read_pickle('experiments/prestates/data.pkl')
eeg_df['P4'] *= 1e6
info_df = pd.read_pickle('experiments/prestates/info.csv')


subj_slopes = []
for subj_id in info_df.subj_id.unique():
    print(subj_id)
    alpha_band = info_df.query('subj_id=={}'.format(subj_id))[['band_l', 'band_r']].values[0]

    bands = [np.array([-3, 0])+alpha_band[0], alpha_band, np.array([0, 15])+alpha_band[1]]
    main_band_ind = 1

    # freq, sxx = sg.welch(x, fs, nperseg=fs, nfft=fs*10)
    # print(band)


    exp_transitions = []
    for block_number in range(15):
        x = eeg_df.query('subj_id=={} & block_number =={}'.format(subj_id, block_number))['P4']
        ov_factor = 2

        f, t, sxx = sg.spectrogram(x, fs, nperseg=fs, noverlap=fs-fs//ov_factor, nfft=10*fs)

        states = np.vstack([sxx[(f >= band[0]) & (f<band[1])].mean(0) for band in bands])
        thresholds = [[np.percentile(x, 33), np.percentile(x, 66)] for x in states]
        states = np.vstack([np.sum([(state>th)*1 for th in threshold], 0) for state, threshold in zip(states, thresholds)])


        all_transitions = []
        for band_ind in range(len(bands)):
            transitions = np.zeros((3, 3))
            a = states[band_ind][:-ov_factor]*10 + states[main_band_ind][ov_factor:]
            for j_from in range(3):
                for j_to in range(3):
                    transitions[j_from, j_to] = np.sum(a==j_from*10+j_to)

            all_transitions.append(transitions/np.sum(transitions, 1)[:, None])

        exp_transitions.append(np.vstack(all_transitions))

    exp_transitions = np.array(exp_transitions)

    slopes = np.zeros(exp_transitions[0].shape)
    for j in range(slopes.shape[0]):
        for k in range(slopes.shape[1]):
            slopes[j, k] = linregress(np.arange(15), exp_transitions[:, j, k]).slope

    subj_slopes.append(slopes)

subj_slopes = np.array(subj_slopes)
pvals = np.zeros(subj_slopes[0].shape)
zscores = np.zeros(subj_slopes[0].shape)
real = subj_slopes[info_df.fb_type.values == 'FB0']
mock = subj_slopes[info_df.fb_type.values == 'FBMock']
for j in range(subj_slopes[0].shape[0]):
    for k in range(subj_slopes[0].shape[1]):

        res = ttest_ind(real[:, j, k], mock[:, j, k])
        pvals[j, k] = res.pvalue
        zscores[j, k] = res.statistic


import seaborn as sns
sns.heatmap(zscores, mask=pvals>0.05, cmap='seismic', square=True, xticklabels=['low Alpha', 'mid Alpha', 'high Alpha'],
            yticklabels=['low Theta', 'mid Theta', 'high Theta', 'low Alpha', 'mid Alpha', 'high Alpha', 'low Beta',
                         'mid Beta', 'high Beta'], center=0)


real_mean = real.mean(0)
mock_mean = mock.mean(0)

sns.heatmap(real_mean, cmap='seismic', square=True, xticklabels=['low Alpha', 'mid Alpha', 'high Alpha'],
            yticklabels=['low Theta', 'mid Theta', 'high Theta', 'low Alpha', 'mid Alpha', 'high Alpha', 'low Beta',
                         'mid Beta', 'high Beta'], center=0)
plt.title('FB0')
plt.subplots_adjust(bottom=0.2)
plt.savefig('FB0transitions.png')
plt.close()

sns.heatmap(mock_mean, cmap='seismic', square=True, xticklabels=['low Alpha', 'mid Alpha', 'high Alpha'],
            yticklabels=['low Theta', 'mid Theta', 'high Theta', 'low Alpha', 'mid Alpha', 'high Alpha', 'low Beta',
                         'mid Beta', 'high Beta'], center=0)
plt.title('FBMock')
plt.subplots_adjust(bottom=0.2)
plt.savefig('FBMocktransitions.png')
plt.close()
