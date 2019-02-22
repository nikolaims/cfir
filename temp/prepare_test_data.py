import pandas as pd
import h5py
import os


orig_data_path = '/home/kolai/Data/alpha_delay'
target_df_path = '/home/kolai/Data/cfir'

channels_to_select = ['P4', 'O2', 'Fp2']
df = pd.DataFrame(columns=['subj', 'block_name'] + channels_to_select)
for exp_name in os.listdir(orig_data_path)[:]:
    h5file_path = '{}/{}/experiment_data.h5'.format(orig_data_path, exp_name)
    with h5py.File(h5file_path) as f:
        subj = int(exp_name.split('subj-')[1].split('_')[0])
        fs = int(f['stream_info.xml'][0].split('nominal_srate>')[1].split('<')[0])
        channels = list(map(lambda x: x.split('<')[0], f['stream_info.xml'][0].split('label>')))[1::2]
        print('\nsubj: {:2d}; fs: {:4d}; exp_name: {}'.format(subj, fs, exp_name))
        for k in [1, 2, 4]:
            label = f['protocol{}'.format(k)].attrs['name']
            data = f['protocol{}/raw_data'.format(k)][:][:, [channels.index(ch) for ch in channels_to_select]]
            data_df = pd.DataFrame(data[:fs * 60], columns=channels_to_select)
            data_df['subj'] = subj
            data_df['block_name'] = label
            df = pd.concat([df, data_df], ignore_index=True, sort=False)
            print('\t block: {:10s}; len: {}'.format(label, len(data_df)))


df.to_pickle('{}/eeg_probes.pkl.zip'.format(target_df_path), compression='gzip')
df2 = pd.read_pickle('{}/eeg_probes.pkl.zip'.format(target_df_path), compression='gzip')
print(df.equals(df2))