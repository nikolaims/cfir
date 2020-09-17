import numpy as np
from scipy.linalg import inv

class ColoredMeasurementNoiseKF:
    """
    Based on paper:
        Chang, G. On kalman filter for linear system with colored measurement noise. J Geod 88, 1163–1170 (2014).
        https://doi.org/10.1007/s00190-014-0751-7
    """
    def __init__(self, n_x, n_z, F, Q, H, Psi, R):
        self.F = F
        self.Q = Q
        self.H = H
        self.Psi = Psi
        self.R = R

        self.x = np.zeros(n_x)
        self.P = np.zeros((n_x, n_x))

        self.ym1 = np.zeros(n_z)


    def step(self, y):
        F = self.F
        H = self.H
        Psi = self.Psi
        xm1 = self.x.copy()
        Pm1 = self.P.copy()

        x_pre = F @ self.x
        P = F @ Pm1 @ F.T + self.Q

        z = y - Psi * self.ym1
        n = z - H.dot(x_pre) + Psi * H.dot(xm1)
        Sigma = H @ P @ H.T + Psi * H @ Pm1 @ H * Psi + self.R - H @ F @ Pm1 @ H.T * Psi - Psi * H @ Pm1 @ F.T @ H.T
        Pxn = P @ H.T - F @ Pm1 @ H.T * Psi
        K = Pxn / Sigma

        self.x = x_pre + K * n
        self.P = P - (K * Sigma)[:, None] @ K[None, :]
        self.ym1 = y
        return x_pre.copy(), self.x.copy(), n, K, Sigma, self.P


class SimpleKF:
    def __init__(self, n_x, n_z, F, Q, H,  R):
        self.F = F
        self.Q = Q
        self.H = H
        self.R = R

        self.x = np.zeros(n_x)
        self.P = np.zeros((n_x, n_x))

    def step(self, z):
        F = self.F
        H = self.H

        x_pre = F @ self.x
        P = F @ self.P @ F.T + self.Q

        y = z - H.dot(x_pre)
        S = H @ P @ H.T + self.R

        K = P @ H / S
        self.x = x_pre + K * y
        self.P = (np.eye(len(self.P)) - K[:, None].dot(H[None, :])).dot(P)
        return x_pre.copy(), self.x.copy(), y, K, S, P


class FixedLagKF:
    """
    Based on code from filterpy package:
        https://github.com/rlabbe/filterpy/blob/master/filterpy/kalman/fixed_lag_smoother.py
    """
    def __init__(self, kf, N):
        self.kf = kf
        self.N = N
        self.count = 0
        self.xSmooth = []

    def step(self, z):
        x_pre, x, y, K, S, P = self.kf.step(z)

        self.xSmooth.append(x_pre)
        # compute invariants
        HTSI = self.kf.H.T / S
        F_LH = (self.kf.F - K[:, None] @ self.kf.H[None, :]).T

        if self.count >= self.N:
            PS = P.copy()  # smoothed P for step i
            for i in range(self.N):
                K = PS @ HTSI  # smoothed gain
                PS = PS @ F_LH  # smoothed covariance

                si = self.count - i
                self.xSmooth[si] = self.xSmooth[si] + K * y
        else:
            self.xSmooth[self.count] = x.copy()

        self.count += 1


