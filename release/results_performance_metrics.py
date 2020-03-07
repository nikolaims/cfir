"""
Figure 2 and 3:  Four performance metrics vs incurred processing delay.
"""

import pandas as pd
import numpy as np
import pylab as plt
import seaborn as sns

from release.constants import DELAY_RANGE, N_SAMPLES_TRAIN, N_SAMPLES_TEST
from release.utils import delay_align
from tqdm import tqdm




# compute metrics: env correlation, phase0 bias, phase0 var
def corr_delay(x, y, delay, bias=0):
    # eff_cor = range(-200, 200, 10)[np.argmax([np.corrcoef(np.roll(np.abs(x), -d), np.abs(y))[0,1] for d in range(-200, 200, 10)])]/500*1000
    x, y = delay_align(x, y, delay)
    corr = np.corrcoef(np.abs(x), np.abs(y))[0, 1]
    phase_zero_moments = np.diff((np.angle(x) >= 0 - bias/360*2*np.pi).astype(int))>0
    phase_bias = np.angle(y)[1:][phase_zero_moments].mean() / 2 / np.pi * 360
    phase_disp = (((np.angle(y)[1:][phase_zero_moments] / 2 / np.pi * 360 - phase_bias)**2).mean())**0.5
    return corr, phase_bias, phase_disp


# load eeg
eeg_df = pd.read_pickle('data/rest_state_probes_real.pkl')

# utils
methods = ['cfir', 'rlscfir', 'wcfir', 'rect', 'ffiltar']
colors = ['#0099d8', '#84BCDA', '#FE4A49', '#A2A79E', '#444444']
metrics = ['corr', 'phase_bias', 'phase_abs_bias', 'phase_disp']
slices = dict([('train', slice(None, N_SAMPLES_TRAIN)), ('test', slice(N_SAMPLES_TRAIN, N_SAMPLES_TRAIN + N_SAMPLES_TEST))])
columns=['method', 'dataset', 'snr', 'delay', 'metric', 'train', 'test', 'params']

# best stats df
stats_df = pd.DataFrame(columns=columns, dtype=float)


# iterate over methods
for j_method, method_name in enumerate(methods):

    # load grid results and params
    res = np.load('results/{}.npy'.format(method_name))
    kwargs_df = pd.read_csv('results/{}_kwargs.csv'.format(method_name))

    # iterate datasets
    for dataset in tqdm(eeg_df['subj_id'].unique(), method_name):

        # query dataset data
        df = eeg_df.query('subj_id == {}'.format(dataset))

        # get snr, sim flag and y_true
        snr = df['snr'].values[0]
        y_true = df['an_signal'].values

        # iterate delays
        for delay in DELAY_RANGE:
            best_corr_dict = (None, None, None)
            best_bias_dict = (None, None, None)
            best_var_dict = (None, None, None)

            # iterate params grid
            for params in kwargs_df.query('subj_id == {} & delay=={}'.format(dataset, delay)).itertuples():
                params = params._asdict()
                y_pred = res[params.pop('Index')]
                assert params.pop('subj_id') == dataset
                assert params.pop('delay') == delay

                # train scores
                train_corr, train_bias, train_var = corr_delay(y_pred[slices['train']], y_true[slices['train']], delay)

                # if one of scores is better continue
                if ((train_corr > (best_corr_dict[0] or 0)) or
                    (abs(train_bias) < abs(best_bias_dict[0] or 1000)) or
                    (train_var < (best_var_dict[0] or 1000))):

                    # compute test scores
                    test_corr, test_bias, test_var = corr_delay(y_pred[slices['test']], y_true[slices['test']],
                                                                delay, train_bias)

                    # if train env CORR is better save train test and params
                    if train_corr > (best_corr_dict[0] or 0):
                        best_corr_dict = (train_corr, test_corr, params)

                    # if train BIAS is better save train test and params
                    if abs(train_bias) < abs(best_bias_dict[0] or 1000):
                        best_bias_dict = (train_bias, test_bias, params)

                    # if train VAR is better save train test and params
                    if train_var < (best_var_dict[0] or 1000):
                        best_var_dict = (train_var, test_var, params)

            # save stats
            stats_dict = {'method': method_name, 'subj_id': dataset, 'snr': snr, 'delay': delay*2,
                          'metric': metrics,
                          'train':  [best_corr_dict[0], best_bias_dict[0],
                                     abs(best_bias_dict[0]) if best_bias_dict[0] else np.nan, best_var_dict[0]],
                          'test':   [best_corr_dict[1], best_bias_dict[1],
                                     abs(best_bias_dict[1]) if best_bias_dict[1] else np.nan, best_var_dict[1]],
                          'params': [best_corr_dict[2], best_bias_dict[2], best_bias_dict[2], best_var_dict[2]]}
            stats_df = stats_df.append(pd.DataFrame(stats_dict), ignore_index=True)

# update dtypes
stats_df['train'] = stats_df['train'].astype(float)
stats_df['test'] = stats_df['test'].astype(float)
stats_df['delay'] = stats_df['delay'].astype(int)
stats_df['row'] = stats_df['metric'].isin(['phase_bias', 'phase_abs_bias'])
stats_df['col'] = stats_df['metric'].isin(['phase_disp', 'phase_abs_bias'])

# save stats df
stats_df.to_pickle('results/stats.pkl')


# plot metrics trade-off
sns.set_style()
g = sns.catplot('delay', 'test', 'method', data=stats_df.fillna(np.inf), col='col', sharey='none', kind='point', dodge=0.2,
                palette=sns.color_palette(colors), linewidth=0, edgecolor='#CCCCCC', height=4, aspect=1, row='row')
def setup_axes(g, xlabel='Delay, ms'):
    # [ax.axvline(2, color='k', alpha=0.5, linestyle='--', zorder=-1000) for ax in g.axes.flatten()]
    g.axes[1, 0].axhline(0, color='k', alpha=0.5, linestyle=':', zorder=-1000)
    g.axes[1, 1].axhline(0, color='k', alpha=0.5, linestyle=':', zorder=-1000)
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    g.axes[0, 0].set_ylim(0, 1)
    g.axes[1, 1].set_ylim(-5, 10)
    g.axes[1, 0].set_ylim(-5, 10)
    g.axes[0, 1].set_ylim(0, 100)
    g.axes[0,0].set_ylabel('$r_a$')
    g.axes[0,1].set_ylabel('$\sigma_\phi$')
    g.axes[1, 0].set_ylabel('$b_\phi$')
    g.axes[1, 1].set_ylabel('$|b_\phi|$')
    g.axes[1,0].set_xlabel(xlabel)
    g.axes[1,1].set_xlabel(xlabel)
    for k, j in [(0,1), (1,0), (1,1)]:
        g.axes[k, j].set_yticklabels(['${:n}^\circ$'.format(x) for x in g.axes[k, j].get_yticks()])
    # g.axes[0,1].set_yticklabels(['${:n}^\circ$'.format(x) for x in g.axes[0, 2].get_yticks()])
    g.axes[0, 0].set_title('A. Envelope corr.')
    g.axes[0, 1].set_title('B. Phase SD')
    g.axes[1, 0].set_title('C. Phase bias ')
    g.axes[1, 1].set_title('D. Phase abs. bias.')
setup_axes(g)
# plt.savefig('results/viz/res-metrics.png', dpi=500)

# plot zero delay metrics vs SNR
g = sns.lmplot('snr', 'test', hue='method', data=stats_df.query('delay==0'), col='col', sharey='none',
               palette=sns.color_palette(colors), height=4, aspect=1, ci=None, row='row')
g.axes[0,0].set_xlim(stats_df.snr.min()-0.1, stats_df.snr.max()+0.1)

g.axes[0,1].lines[methods.index('rect')].set_alpha(0)
g.axes[1,0].lines[methods.index('rect')].set_alpha(0)
g.axes[1,1].lines[methods.index('rect')].set_alpha(0)

setup_axes(g, 'SNR')
# plt.savefig('results/viz/res-metrics-delay0.png', dpi=500)