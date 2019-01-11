import numpy as np
from pycfir.filters import get_x_chirp, RectEnvDetector
import pylab as plt

fs = 500
x, amp = get_x_chirp(fs)
x += np.random.normal(size=len(x))*0.2
n_taps_bandpass_list = np.arange(20, 500, 10)
n_taps_smooth_list = np.arange(20, 500, 10)


corrs = np.zeros((len(n_taps_bandpass_list), len(n_taps_smooth_list)))
delays = np.zeros((len(n_taps_bandpass_list), len(n_taps_smooth_list)))
x_coord = np.zeros((len(n_taps_bandpass_list), len(n_taps_smooth_list)))
y_coord = np.zeros((len(n_taps_bandpass_list), len(n_taps_smooth_list)))
for j_bandpass, n_taps_bandpass in enumerate(n_taps_bandpass_list):
    for j_smooth, n_taps_smooth in enumerate(n_taps_smooth_list):

        n_taps = [n_taps_bandpass, n_taps_smooth]
        filt = RectEnvDetector([8, 12], fs, *n_taps)
        y = filt.apply(x)[sum(n_taps)//2:]
        y_true = amp[:-sum(n_taps)//2]
        corr = np.corrcoef(y, y_true)[0, 1]
        corrs[j_bandpass, j_smooth] = 0 if np.isnan(corr) else corr
        delays[j_bandpass, j_smooth] = sum(n_taps)//2
        x_coord[j_bandpass, j_smooth] = n_taps_smooth
        y_coord[j_bandpass, j_smooth] = n_taps_bandpass


fig, ax = plt.subplots(1,2, dpi=200)
im = ax[0].scatter(x_coord.flatten(), y_coord.flatten(), (1-delays.flatten()/delays.max())*100+5, corrs.flatten(), cmap='viridis', picker=True, vmin=np.quantile(corrs, 0.3))
ax[0].set_xlabel('n_taps_smoother')
ax[0].set_ylabel('n_taps_bandpass_filter')
#plt.colorbar()
fig.colorbar(im, ax=ax[0])

def onpick3(event):
    ind = event.ind
    ax[1].clear()

    print('onpick3 scatter:', ind, np.take(x_coord.flatten(), ind), np.take(y_coord.flatten(), ind))
    n_taps_smooth = np.take(x_coord.flatten(), ind).astype(int)[0]
    n_taps_bandpass = np.take(y_coord.flatten(), ind).astype(int)[0]
    n_taps = [n_taps_bandpass, n_taps_smooth]
    print(n_taps)
    filt = RectEnvDetector([8, 12], fs, *n_taps)
    y = filt.apply(x)[sum(n_taps) // 2:]
    y_true = amp[:-sum(n_taps) // 2]

    ax[1].plot(np.arange(len(y_true))/fs, y_true/max(y_true))
    ax[1].plot(np.arange(len(y)) / fs, y/max(y))
    ax[1].set_xlim(0, 10)
    ax[1].legend(['GND', 'corr. = {:.3f}\ndelay = {:.0f}ms'.format(np.corrcoef(y, y_true)[0, 1], sum(n_taps)/2/fs*1000)])
    plt.draw()

fig.canvas.mpl_connect('pick_event', onpick3)

plt.tight_layout()
plt.show()