import pandas as pd
import numpy as np
from settings import DELAY_RANGE, N_SAMPLES_TRAIN, N_SAMPLES_TEST
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

def corr_delay(x, y, delay, metric='corr'):
    x, y = delay_align(x, y, delay)
    if metric == 'corr':
        return np.corrcoef(np.abs(x), np.abs(y))[0, 1]
    if metric == 'phase_bias':
        return np.angle(y)[1:][np.diff((np.angle(x) >= 0).astype(int))>0].mean()/2/np.pi*360
    if metric == 'phase_std':
        return np.median(np.abs(np.angle(y)[1:][np.diff((np.angle(x) >= 0).astype(int)) > 0]))/2/np.pi*360
    raise NameError('Bad metric name')



eeg_df = pd.read_pickle('data/rest_state_probes.pkl')
metric_name = 'phase_bias'
better = lambda x, y: (x > (y or 0)) if metric_name=='corr' else (np.abs(x) < np.abs(y or 10000))

methods = ['cfir', 'rlscfir', 'ffiltar', 'whilbert', 'rect']

stats_df = pd.DataFrame(columns=['method', 'dataset', 'snr', 'sim', 'delay', 'max_corr_train', 'method_corr_train', 'max_corr_test',
                                 'method_corr_test', ])

methods = ['ffiltar']
for j_method, method_name in enumerate(methods):

    res = np.load('results/{}.npy'.format(method_name))
    kwargs_df = pd.read_csv('results/{}_kwargs.csv'.format(method_name))



    for dataset in tqdm(eeg_df['dataset'].unique()):
        df = eeg_df.query('dataset == "{}"'.format(dataset))
        snr = df['snr'].values[0]
        sim = df['sim'].values[0]
        y_true = df['an_signal'].values
        max_corr = [None, None, None]
        for delay in DELAY_RANGE:
            method_corr = [None, None, None]
            for ind in kwargs_df.query('dataset == "{}"'.format(dataset)).itertuples():
                y_pred = res[ind.Index]
                if method_name == 'ffiltar':
                    y_pred = np.roll(y_pred, 1)
                corr_train = corr_delay(y_pred[:N_SAMPLES_TRAIN], y_true[:N_SAMPLES_TRAIN], delay, metric=metric_name)
                corr_test = corr_delay(y_pred[N_SAMPLES_TRAIN:N_SAMPLES_TRAIN + N_SAMPLES_TEST], y_true[N_SAMPLES_TRAIN:N_SAMPLES_TRAIN + N_SAMPLES_TEST], delay, metric=metric_name)
                if ind.delay == delay and better(corr_train, method_corr[0]): method_corr = (corr_train, corr_test, ind)
                if better(corr_train, max_corr[0]): max_corr = (corr_train, corr_test, ind)

            stats_dict = {'method': method_name, 'dataset': dataset, 'snr': snr, 'sim': sim ,'delay': delay, 'max_corr_train': max_corr[0], 'method_corr_train': method_corr[0],
             'max_corr_test': max_corr[1], 'method_corr_test': method_corr[1]}
            if max_corr[2] is not None: stats_dict.update(dict([('max_'+key, val) for key, val in max_corr[2]._asdict().items()]))
            if method_corr[2] is not None: stats_dict.update(dict([('method_' + key, val) for key, val in method_corr[2]._asdict().items()]))
            stats_df = stats_df.append(stats_dict, ignore_index=True)



stats_df['snr_type'] = 'low'
stats_df['snr_type'][stats_df['snr']>=stats_df['snr'].median()] = 'high'
stats_df['period'] = 'next'
stats_df.loc[stats_df['delay']>=0, 'period'] = 'last'
stats_df.loc[stats_df['delay']>=50, 'period'] = 'before last'
stats_df['sim_type'] = 'real'
stats_df['sim_type'][stats_df['sim'].astype(bool)] = 'sim'


g = sns.catplot('period', 'method_corr_test', hue='method', data=stats_df.query('sim==0'), ci='sd', kind='bar', col='snr_type',  order=['before last', 'last', 'next'])
[ax.axvline(1.5, color='k', alpha=0.8, linestyle='--') for ax in g.axes.flatten()]
#plt.semilogx()