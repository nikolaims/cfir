import pandas as pd
import numpy as np
from release.constants import DELAY_RANGE, N_SAMPLES_TRAIN, N_SAMPLES_TEST
from tqdm import tqdm
import pylab as plt
import seaborn as sns


def delay_align(x, y, delay):
    if delay >= 0:
        x = x[delay:]
        y = y[:-delay or None]
    else:
        x = x[:delay]
        y = y[abs(delay):]
    return x, y


def corr_delay(x, y, delay):
    x, y = delay_align(x, y, delay)
    corr = np.corrcoef(np.abs(x), np.abs(y))[0, 1]
    phase = np.abs(np.mean(x/np.abs(x)*np.conj(y)/np.abs(y)))
    # phase = np.angle(y)[1:][np.diff((np.angle(x) >= 0).astype(int))>0].mean() / 2 / np.pi * 360
    # phase = np.abs(np.angle(y)[1:][np.diff((np.angle(x) >= 0).astype(int)) > 0] / 2 / np.pi * 360 - phase).mean()
    return corr, phase


eeg_df = pd.read_pickle('data/rest_state_probes_real.pkl')

methods = ['cfir', 'rlscfir', 'rect']


columns = ['method', 'dataset', 'snr', 'sim', 'delay', 'metric', 'train', 'test', 'params']
dtypes = ['str', 'str', 'float', 'int', 'int', 'str', 'float', 'float', 'object']

stats_df = pd.DataFrame(columns=columns, dtype=float)

slices = dict([('train', slice(None, N_SAMPLES_TRAIN)), ('test', slice(N_SAMPLES_TRAIN, N_SAMPLES_TRAIN + N_SAMPLES_TEST))])


for j_method, method_name in enumerate(methods):
    res = np.load('results/{}.npy'.format(method_name))
    kwargs_df = pd.read_csv('results/{}_kwargs.csv'.format(method_name))

    for dataset in tqdm(eeg_df['dataset'].unique(), method_name):
        df = eeg_df.query('dataset == "{}"'.format(dataset))
        snr = df['snr'].values[0]
        sim = df['sim'].values[0]
        y_true = df['an_signal'].values

        for delay in DELAY_RANGE:
            best_corr_dict = (None, None, None)
            best_phase_dict = (None, None, None)
            for params in kwargs_df.query('dataset == "{}" & delay=={}'.format(dataset, delay)).itertuples():
                params = params._asdict()
                y_pred = res[params.pop('Index')]
                assert params.pop('dataset') == dataset
                assert params.pop('delay') == delay
                if method_name == 'ffiltar': y_pred = np.roll(y_pred, 1)
                if method_name == 'whilbert': y_pred = np.roll(y_pred, -1)
                if method_name == 'cfir': y_pred = np.roll(y_pred, 0)
                if method_name == 'rlscfir': y_pred = np.roll(y_pred, 0)
                train_corr, train_phase = corr_delay(y_pred[slices['train']], y_true[slices['train']], delay)
                if (train_corr > (best_corr_dict[0] or 0)) or (train_phase > (best_phase_dict[0] or 0)):
                    test_corr, test_phase = corr_delay(y_pred[slices['test']], y_true[slices['test']], delay)
                    if train_corr > (best_corr_dict[0] or 0):
                        best_corr_dict = (train_corr, test_corr, params)
                    if train_phase > (best_phase_dict[0] or 0):
                        best_phase_dict = (train_phase, test_phase, params)


            stats_dict = {'method': method_name, 'dataset': dataset, 'snr': snr, 'sim': sim, 'delay': delay,
                          'metric': ['corr', 'phase'], 'train': [best_corr_dict[0], best_phase_dict[0]],
                          'test': [best_corr_dict[1], best_phase_dict[1]], 'params': [best_corr_dict[2], best_phase_dict[2]]}

            stats_df = stats_df.append(pd.DataFrame(stats_dict), ignore_index=True)

stats_df['train'] = stats_df['train'].astype(float)
stats_df['test'] = stats_df['test'].astype(float)
stats_df.to_pickle('results/stats.pkl')

stats_df = pd.read_pickle('results/stats.pkl')
stats_df['snr_cat'] = stats_df['snr'].apply(lambda x: 'High' if x> stats_df['snr'].median() else 'Low')

stats_df['delay_cat'] = stats_df['delay'].astype(int)#.apply(lambda x: '[100, 200)' if x>=50 else ('[0, 100)' if x>=0 else '[-100, 0)'))


#flatui = ["#F15152", "#11A09E", "#91BFBE"]
flatui = ['#0099d8', '#84BCDA', '#FE4A49', '#A2A79E', '#CCDAD1']
#sns.set(rc={'figure.figsize': (5,5)})
g = sns.catplot('delay_cat', 'test', 'method', data=stats_df, col='snr_cat', row='metric', sharey='row', kind='bar', palette=sns.color_palette(flatui), linewidth=0, edgecolor='#CCCCCC', height=2.5, aspect=1.5, col_order=['Low', 'High'])
[ax.axvline(0.5, color='k', alpha=0.5, linestyle='--') for ax in g.axes.flatten()]
g.axes[0,0].set_ylabel('$r_a$')
g.axes[1,0].set_ylabel('$b_\phi$')
g.axes[1,0].set_xlabel('Delay range, ms')
g.axes[1,1].set_xlabel('Delay range, ms')
#g.axes[1,0].set_yticks([0, 5, 7.2, 10, 14.4])
#g.axes[1,0].set_yticklabels(['$0^\circ$\nn=0', '$5^\circ$', 'n=1', '$10^\circ$', 'n=2'])
g.axes[0,0].set_title('Low SNR')
g.axes[0,1].set_title('High SNR')
g.axes[1,0].set_title('')
g.axes[1,1].set_title('')
#for j in range(2): [g.axes[1,j].axhline(k*7.2, color='k', alpha=0.1, linewidth=1, zorder=-100) for k in range(10)]

plt.savefig('results/viz/res-metrics.png', dpi=200)


g = sns.relplot('delay', 'test', 'method', data=stats_df.query('dataset=="alpha2-delay-subj-21_12-06_12-15-09"'), row='metric', facet_kws=dict(sharey='row'), kind='line', palette=sns.color_palette(flatui),  height=2.5, aspect=1.5, col_order=['Low', 'High'])
