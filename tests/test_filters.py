from pycfir.filters import *

def get_x_chirp():
    fs = 500
    t = np.arange(fs * 60) / fs
    randx = np.random.randn(len(t))
    x = sg.chirp(t, f0=8, f1=10, t1=60, method='linear') * sg.filtfilt(*sg.butter(4, 2 / fs * 2, 'low'), randx)
    return x

def test_sw_filter1():
    x = get_x_chirp()
    y = rt_emulate(SquareSWFilter(1), x)
    assert np.allclose(x ** 2, y) == True


def test_sw_filter2():
    x = get_x_chirp()
    y = rt_emulate(SquareSWFilter(500), x, 8)
    assert np.allclose(x[7::8] ** 2, y[7::8]) == True
