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
    corr = np.corrcoef(np.abs(x), np.abs(y))[0, 1]
    phase = np.angle(y)[1:][np.diff((np.angle(x) >= 0).astype(int))>0].mean() / 2 / np.pi * 360
    return corr, phase


eeg_df = pd.read_pickle('data/rest_state_probes_real.pkl')

methods = ['cfir', 'rlscfir', 'ffiltar', 'whilbert', 'rect']


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
                #if method_name == 'ffiltar': y_pred = np.roll(y_pred, 1)

                train_corr, train_phase = corr_delay(y_pred[slices['train']], y_true[slices['train']], delay)
                if (train_corr > (best_corr_dict[0] or 0)) or (np.abs(train_phase) < np.abs(best_phase_dict[0] or 360)):
                    test_corr, test_phase = corr_delay(y_pred[slices['test']], y_true[slices['test']], delay)
                    if train_corr > (best_corr_dict[0] or 0):
                        best_corr_dict = (train_corr, test_corr, params)
                    if np.abs(train_phase) < np.abs(best_phase_dict[0] or 360):
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

stats_df['delay_cat'] = stats_df['delay'].apply(lambda x: '100:200ms' if x>=50 else ('0:100ms' if x>=0 else '0:-100ms'))


flatui = ["#F15152", "#11A09E", "#91BFBE"]

sns.catplot('method', 'test', 'delay_cat', data=stats_df, col='snr_cat', row='metric', sharey='row', kind='bar', palette=sns.color_palette(flatui))