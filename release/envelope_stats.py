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
    #phase = np.abs(np.mean(x/np.abs(x)*np.conj(y)/np.abs(y)))
    phase_zero_moments = np.diff((np.angle(x) >= 0).astype(int))>0
    phase_bias = np.angle(y)[1:][phase_zero_moments].mean() / 2 / np.pi * 360
    phase_disp = (((np.angle(y)[1:][phase_zero_moments] / 2 / np.pi * 360 - phase_bias)**2).mean())**0.5
    return corr, phase_bias, phase_disp


ONLY_ZERO_DELAY = False

eeg_df = pd.read_pickle('data/rest_state_probes_real.pkl')

methods = ['cfir', 'rlscfir', 'wcfir', 'rect', 'ffiltar'][:None if ONLY_ZERO_DELAY else -1]


columns = ['method', 'dataset', 'snr', 'sim', 'delay', 'metric', 'train', 'test', 'params']
dtypes = ['str', 'str', 'float', 'int', 'int', 'str', 'float', 'float', 'object']

stats_df = pd.DataFrame(columns=columns, dtype=float)

slices = dict([('train', slice(None, N_SAMPLES_TRAIN)), ('test', slice(N_SAMPLES_TRAIN, N_SAMPLES_TRAIN + N_SAMPLES_TEST))])

if ONLY_ZERO_DELAY: DELAY_RANGE = np.array([0])

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
            best_phase_bias_dict = (None, None, None)
            best_phase_disp_dict = (None, None, None)
            for params in kwargs_df.query('dataset == "{}" & delay=={}'.format(dataset, delay)).itertuples():
                params = params._asdict()
                y_pred = res[params.pop('Index')]
                assert params.pop('dataset') == dataset
                assert params.pop('delay') == delay
                # if method_name == 'ffiltar': y_pred = np.roll(y_pred, 5)
                train_corr, train_phase_bias, train_phase_disp = corr_delay(y_pred[slices['train']], y_true[slices['train']], delay)
                if (train_corr > (best_corr_dict[0] or 0)) or (train_phase_bias < (best_phase_bias_dict[0] or 1000)) or (train_phase_disp < (best_phase_disp_dict[0] or 1000)):
                    test_corr, test_phase_bias, test_phase_disp = corr_delay(y_pred[slices['test']], y_true[slices['test']], delay)
                    if train_corr > (best_corr_dict[0] or 0):
                        best_corr_dict = (train_corr, test_corr, params)
                    if train_phase_bias < (best_phase_bias_dict[0] or 1000):
                        best_phase_bias_dict = (train_phase_bias, test_phase_bias, params)

                    if train_phase_bias < (best_phase_disp_dict[0] or 1000):
                        best_phase_disp_dict = (train_phase_disp, test_phase_disp, params)


            stats_dict = {'method': method_name, 'dataset': dataset, 'snr': snr, 'sim': sim, 'delay': delay,
                          'metric': ['corr', 'phase_bias', 'phase_disp'], 'train': [best_corr_dict[0], best_phase_bias_dict[0], best_phase_disp_dict[0]],
                          'test': [best_corr_dict[1], best_phase_bias_dict[1], best_phase_disp_dict[1]], 'params': [best_corr_dict[2], best_phase_bias_dict[2], best_phase_disp_dict[2]]}

            stats_df = stats_df.append(pd.DataFrame(stats_dict), ignore_index=True)

stats_df['train'] = stats_df['train'].astype(float)
stats_df['test'] = stats_df['test'].astype(float)
# stats_df.to_pickle('results/stats.pkl')

# stats_df = pd.read_pickle('results/stats.pkl')
stats_df['snr_cat'] = stats_df['snr'].apply(lambda x: 'High' if x> stats_df['snr'].median() else 'Low')

stats_df['delay_cat'] = stats_df['delay'].astype(int)#.apply(lambda x: '[100, 200)' if x>=50 else ('[0, 100)' if x>=0 else '[-100, 0)'))


#flatui = ["#F15152", "#11A09E", "#91BFBE"]
flatui = ['#0099d8', '#84BCDA', '#FE4A49', '#A2A79E', '#444444',]
#sns.set(rc={'figure.figsize': (5,5)})
g = sns.catplot('delay_cat', 'test', 'method', data=stats_df, col='metric', sharey='col', kind='point', dodge=True, palette=sns.color_palette(flatui), linewidth=0, edgecolor='#CCCCCC', height=3, aspect=1)
def setup_axes(g, xlabel='Delay range, ms'):
    [ax.axvline(2, color='k', alpha=0.5, linestyle='--', zorder=-1000) for ax in g.axes.flatten()]
    plt.subplots_adjust(wspace=0.35)
    g.axes[0,0].set_ylabel('$r_a$')
    g.axes[0,1].set_ylabel('$b_\phi$')
    g.axes[0,2].set_ylabel('$\sigma_\phi$')
    g.axes[0,0].set_xlabel(xlabel)
    g.axes[0,1].set_xlabel(xlabel)
    g.axes[0,2].set_xlabel(xlabel)
    g.axes[0,1].set_yticklabels(['${:n}^\circ$'.format(x) for x in g.axes[0, 1].get_yticks()])
    g.axes[0,2].set_yticklabels(['${:n}^\circ$'.format(x) for x in g.axes[0, 2].get_yticks()])
    g.axes[0,0].set_title('A. Envelope corr.')
    g.axes[0,1].set_title('B. Phase bias ')
    g.axes[0,2].set_title('C. Phase var.')
setup_axes(g)
#for j in range(2): [g.axes[1,j].axhline(k*7.2, color='k', alpha=0.1, linewidth=1, zorder=-100) for k in range(10)]

plt.savefig('results/viz/res-metrics-{}.png'.format('ffiltar' if ONLY_ZERO_DELAY else ''), dpi=500)



g = sns.lmplot('snr', 'test', hue='method', data=stats_df.query('delay==0'), col='metric', sharey='none', palette=sns.color_palette(flatui), height=3, aspect=1, ci=None)

g.axes[0,0].set_xlim(stats_df.snr.min()-0.1, stats_df.snr.max()+0.1)
g.axes[0,0].set_ylim(0, 1)
g.axes[0,1].set_ylim(-20, 30)
g.axes[0,2].set_ylim(30, 90)
g.axes[0,1].lines[-1 - int(ONLY_ZERO_DELAY)].set_alpha(0)
g.axes[0,2].lines[-1 - int(ONLY_ZERO_DELAY)].set_alpha(0)
setup_axes(g, 'SNR')

plt.savefig('results/viz/res-metrics-delay0-{}.png'.format('ffiltar' if ONLY_ZERO_DELAY else ''), dpi=500)