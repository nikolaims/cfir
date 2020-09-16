import numpy as np

class ColoredMeasurementNoiseKF:
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

        x = F @ self.x
        P = F @ Pm1 @ F.T + self.Q

        z = y - Psi * self.ym1
        n = z - H.dot(x) + Psi * H.dot(xm1)
        Sigma = H @ P @ H.T + Psi * H @ Pm1 @ H * Psi + self.R - H @ F @ Pm1 @ H.T * Psi - Psi * H @ Pm1 @ F.T @ H.T
        Pxn = P @ H.T - F @ Pm1 @ H.T * Psi
        K = Pxn / Sigma

        self.x = x + K * n
        self.P = P - (K * Sigma)[:, None] @ K[None, :]
        self.ym1 = y


