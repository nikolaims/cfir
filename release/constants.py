import numpy as np
FS = 500
WELCH_NPERSEG = FS*4
ALPHA_BAND = (8, 12)
N_SAMPLES_TRAIN = 60*FS
N_SAMPLES_TEST = 60*FS
N_SUBJECTS = 10
DELAY_RANGE = np.array([-50, -25, 0, 25, 50, 75, 100, 125])#np.linspace(-50, 100, 16, dtype=int) #-100ms:100ms