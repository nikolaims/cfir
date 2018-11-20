from pycfir.filters import *

class SquareSWFilter(SlidingWindowFilter):
    def process_buffer(self):
        return self.buffer[-1]**2

def test_sw_filter1():
    x, amp = get_x_chirp(500)
    y = rt_emulate(SquareSWFilter(1), x)
    np.testing.assert_allclose(x ** 2, y)


def test_sw_filter2():
    x, amp = get_x_chirp(500)
    y = rt_emulate(SquareSWFilter(500), x, 8)
    np.testing.assert_allclose(x[7::8] ** 2, y[7::8])

def test_filtfilt_sw_filter_identity():
    x, amp = get_x_chirp(500)
    filt = FiltFiltRectSWFilter(1000, ([1., 0], [1]), ([1., 0], [1]), delay=0)
    np.testing.assert_allclose(np.abs(x), rt_emulate(filt, x))