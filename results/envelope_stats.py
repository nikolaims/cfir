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

def corr_delay(x, y, delay):
    x, y = delay_align(x, y, delay)
    corr = np.corrcoef(x, y)[0, 1]
    return corr

eeg_df = pd.read_pickle('data/rest_state_probes.pkl').query('dataset=="alpha2-delay-subj-1_11-06_17-15-29"')


methods = ['cfir', 'rlscfir', 'ffiltar', 'whilbert', 'rect']

cm = sns.color_palette(n_colors=len(methods))
for j_method, method_name in enumerate(methods):

    res = np.load('results/{}.npy'.format(method_name))
    kwargs_df = pd.read_csv('results/{}_kwargs.csv'.format(method_name))


    stats_df = pd.DataFrame(columns=['dataset', 'delay', 'max_corr_train', 'method_corr_train', 'max_corr_test',
                                     'method_corr_test'] + list(map(lambda x: 'max_' + x, kwargs_df.columns)) +
                                    list(map(lambda x: 'method_' + x, kwargs_df.columns)))
    for dataset in tqdm(eeg_df['dataset'].unique()):
        df = eeg_df.query('dataset == "{}"'.format(dataset))
        y_true = df['an_signal'].abs().values
        max_corr = [None, None, None]
        for delay in DELAY_RANGE:
            method_corr = [None, None, None]
            for ind in kwargs_df.query('dataset == "{}"'.format(dataset)).itertuples():
                y = np.abs(res[ind.Index])
                corr_train = corr_delay(y[:N_SAMPLES_TRAIN], y_true[:N_SAMPLES_TRAIN], delay)
                corr_test = corr_delay(y[N_SAMPLES_TRAIN:N_SAMPLES_TRAIN+N_SAMPLES_TEST], y_true[N_SAMPLES_TRAIN:N_SAMPLES_TRAIN+N_SAMPLES_TEST], delay)
                if ind.delay == delay and corr_train > (method_corr[0] or 0): method_corr = (corr_train, corr_test, ind)
                if corr_train > (max_corr[0] or 0): max_corr = (corr_train, corr_test, ind)

            stats_dict = {'dataset': dataset, 'delay': delay, 'max_corr_train': max_corr[0], 'method_corr_train': method_corr[0],
             'max_corr_test': max_corr[1], 'method_corr_test': method_corr[1]}
            if max_corr[2] is not None: stats_dict.update(dict([('max_'+key, val) for key, val in max_corr[2]._asdict().items()]))
            if method_corr[2] is not None: stats_dict.update(dict([('method_' + key, val) for key, val in method_corr[2]._asdict().items()]))
            stats_df = stats_df.append(stats_dict, ignore_index=True)



    plt.plot(stats_df.query('dataset == "alpha2-delay-subj-1_11-06_17-15-29"')['delay'], stats_df.query('dataset == "alpha2-delay-subj-1_11-06_17-15-29"')['max_corr_train'], label=j_method, color=cm[j_method])
    plt.plot(stats_df.query('dataset == "alpha2-delay-subj-1_11-06_17-15-29"')['delay'],
             stats_df.query('dataset == "alpha2-delay-subj-1_11-06_17-15-29"')['method_corr_train'], '--', label=method_name, color=cm[j_method])
    #plt.plot(stats_df.query('dataset == "{}"'.format('sim8'))['delay'], stats_df.query('dataset == "{}"'.format('sim8'))['max_delay'])
plt.legend()