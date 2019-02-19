from spectrum.yulewalker import aryule
from statsmodels.regression.linear_model import yule_walker
import numpy as np
import pylab as plt
from scipy.signal import hilbert

x_func = lambda t: np.sin(10 * 2 * np.pi * t / 500) * np.sin(
    0.91 * 2 * np.pi * t / 500 + np.sin(0.31 * 2 * np.pi * t / 500) * 0.5)

t_train = np.arange(1000)
t_test = np.arange(1000, 1200)
x_train = x_func(t_train)
x_test = x_func(t_test)

order = 50
# ar, p, k = aryule(x_func(t_train), order, norm='biased')
ar, s = yule_walker(x_train, order, 'mle')

pred = x_train.tolist()

for x in range(len(t_test)):
    # pred.append(np.roll(ar, 0)[::-1].dot(pred[-order:]))
    pred.append(ar[::-1].dot(pred[-order:]))
plt.figure(dpi=200)
plt.plot(pred)

plt.plot(t_test, x_test, '--')
plt.plot(t_train, x_train)
# plt.plot(t_train, np.real(hilbert(x_train)))

plt.ylim(-2, 2)
plt.show()
