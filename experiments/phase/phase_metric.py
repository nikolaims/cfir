from pycfir.filters import CFIRBandEnvelopeDetector, get_x_chirp, rt_emulate, FiltFiltARHilbertFilter, AdaptiveCFIRBandEnvelopeDetector
import numpy as np
import pylab as plt
import pickle
from scipy.signal import hilbert

fs = 500
delays = np.arange(-100, 101, 50)
with open('alpha_sim_snr.pkl', 'rb') as handle:
    sim_dict = pickle.load(handle)

n_seconds = 20
snr = 2
x = sim_dict['alpha'][:fs*n_seconds] * snr + sim_dict['noise'][:fs * n_seconds]
angle = np.angle(hilbert(sim_dict['alpha'][:fs*n_seconds]))
amp = sim_dict['envelope'][:fs*n_seconds]

ax = plt.subplot(111)
for j_method, method in enumerate(['cFIR', 'acFIR', 'CHEN']):
    for j, delay in enumerate(delays):
        print(method, delay)
        #filt = WHilbertFilter(500, fs, [8, 12], delay)
        if method == 'cFIR':
            filt = CFIRBandEnvelopeDetector([8,12], fs, delay)
        elif method == 'acFIR':
            filt = AdaptiveCFIRBandEnvelopeDetector([8, 12], fs, delay, n_taps=1000, n_fft=2000, ada_n_taps=500)
        else:
            filt = FiltFiltARHilbertFilter([8, 12], fs, 500, 500, 40, delay, ar_order=100)
        # filt = AdaptiveEnvelopePredictor(filt, 500, -50)
        # filt = FiltFiltARHilbertFilter([8,12], fs, 500, 500, 40, delay, ar_order=100)
        # filt = ARCFIRBandEnvelopeDetector([8,12], fs, delay, delay_threshold=50)



        y_hat = rt_emulate(filt, x, 10)[delay if delay > 0 else None:delay if delay < 0 else None]
        y = x[-delay if delay < 0 else None:-delay if delay > 0 else None]
        y_ang = angle[-delay if delay < 0 else None:-delay if delay > 0 else None]





        h = ((y_ang[1:]))[np.diff((np.abs(np.angle(y_hat) - 0) < 0.1).astype(int))==1]
        ax = plt.subplot(3, len(delays), j+1 + len(delays)*j_method, projection='polar', sharex=ax)
        ax.hist(h, bins=np.arange(361)[::12]/360*np.pi*2 - np.pi, density=True)
        ax.set_title('{}ms\n'.format(delay*1000//fs))

