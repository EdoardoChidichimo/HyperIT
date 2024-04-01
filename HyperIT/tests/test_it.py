import numpy as np
from HyperIT.visualisations.it import *


np.random.seed(42)
n = 1000

A = np.zeros(n)
B = np.zeros(n)
C = np.zeros(n)
D = np.zeros(n)
E = np.zeros(n)
F = np.zeros(n)
A[0], B[0], C[0], D[0], E[0], F[0] = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1

std_dev = 0.1
for t in range(1, n):
    A[t] = np.sin(C[t-1] + E[t-1]) + np.random.normal(0, std_dev)
    B[t] = 0.5 * A[t-1] + np.random.normal(0, std_dev)
    C[t] = 0.3 * B[t-1] + np.exp(-D[t-1]) + np.random.normal(0, std_dev)
    D[t] = D[t-1]**2 - 0.1 * D[t-1] + np.random.normal(0, std_dev)
    E[t] = 0.7 * B[t-1] + np.cos(A[t-1]) + np.random.normal(0, std_dev)
    F[t] = 3 * np.sin(F[t-1]) + np.random.normal(0, std_dev)
    
# def epoch_it(data, n_epochs):
#     if len(data) % n_epochs != 0:
#         raise ValueError("The length of the time series must be divisible by the number of epochs.")
#     epoch_length = len(data) // n_epochs
#     return data.reshape(n_epochs, epoch_length)

# A = epoch_it(A, 10)
# B = epoch_it(B, 10)
# C = epoch_it(C, 10)
# D = epoch_it(D, 10)
# E = epoch_it(E, 10)
# F = epoch_it(F, 10)

### FOR EPOCHED DATA
# data1 = np.stack([A, B, C], axis = 1) #  10, 3, 100 (epo, ch, sample)
# data2 = np.stack([D, E, F], axis = 1) #  10, 3, 100 (epo, ch, sample)

### FOR UNEPOCHED DATA
#data1 = np.vstack([A, B, C]) #  3, 1000 (ch, sample)
#data = np.vstack([D, E, F])  #  3, 1000 (ch, sample)



data = np.vstack([A, B, C, D, E, F])
channel_names = [['A', 'B', 'C', 'D', 'E', 'F'], ['A', 'B', 'C', 'D', 'E', 'F']]

it = HyperIT(data, data, channel_names=channel_names)


#  = [0,0]
# phi_dict_xy, phi_dict_yx = it.compute_atoms(vis=True, plot_channels=plot_channels)

it.compute_mi(estimator_type='symbolic', calc_sigstats=False, vis=True)
it.compute_te(estimator_type='gaussian', calc_sigstats=True, vis=True)

#te_matrix_xy, te_matrix_yx = it.compute_te()
