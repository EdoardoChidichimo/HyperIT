import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

from it import *


import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests


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

# plt.figure(figsize=(12, 8))
# plt.plot(A, label='Unit A')
# plt.plot(B, label='Unit B')
# plt.plot(C, label='Unit C')
# plt.plot(D, label='Unit D')
# plt.plot(E, label='Unit E')
# plt.legend()
# plt.title('States of Information Processing Units Over Time with Noise')
# plt.xlabel('Time')
# plt.ylabel('State')
# plt.show()
    

# data = {
#     'A': A,
#     'B': B,
#     'C': C,
#     'D': D,
#     'E': E
# }


def epoch_it(data, n_epochs):
    if len(data) % n_epochs != 0:
        raise ValueError("The length of the time series must be divisible by the number of epochs.")
    epoch_length = len(data) // n_epochs
    return data.reshape(n_epochs, epoch_length)


A = epoch_it(A, 10)
B = epoch_it(B, 10)
C = epoch_it(C, 10)
D = epoch_it(D, 10)
E = epoch_it(E, 10)
F = epoch_it(F, 10)


    
setup_JIDT(os.getcwd())

eeg_data = np.stack([A, B, C, D, E, F], axis = 1) # EPOCHED 10, 5, 100 (epo, ch, sample)
#eeg_data = np.vstack([A, B, C, D, E, F]) # UNEPOCHED 5, 1000 (ch, sample)


te, sigstats = compute_te(eeg_data, calc_sigstats=True, mode = 'kernel')
plot_it(te, sigstats, False, np.array(['A', 'B', 'C', 'D', 'E', 'F']))




# G = nx.DiGraph()
# G.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F'])

# edges = [('A', 'B', {'interaction': 'linear'}),
#          ('B', 'C', {'interaction': 'linear'}), ('B', 'E', {'interaction': 'linear'}),
#          ('A', 'E', {'interaction': 'non-linear'}), ('D', 'D', {'interaction': 'non-linear'}),
#          ('C', 'A', {'interaction': 'non-linear'}), ('D', 'C', {'interaction': 'non-linear'}),
#          ('E', 'A', {'interaction': 'non-linear'}), ('F', 'F', {'interaction': 'non-linear'})]

# for u, v, interaction in edges:
#     G.add_edge(u, v, interaction=interaction['interaction'])

# pos = nx.kamada_kawai_layout(G)

# linear_edges = [(u, v) for u, v, d in G.edges(data=True) if d['interaction'] == 'linear' and u != v]
# non_linear_edges = [(u, v) for u, v, d in G.edges(data=True) if d['interaction'] == 'non-linear' and not (u == v or {u,v} == {'A', 'E'} or {u,v} == {'E', 'A'})]

# linear_self_loop_edges = [(u, v) for u, v, d in G.edges(data=True) if d['interaction'] == "linear" and u == v]
# non_linear_self_loop_edges = [(u, v) for u, v, d in G.edges(data=True) if d['interaction'] == "non-linear" and u == v and not ({u,v} == {'A', 'E'} or {u,v} == {'E', 'A'})]

# nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightgreen")

# nx.draw_networkx_edges(G, pos, edgelist=linear_edges, 
#                        edge_color="black", arrows=True, arrowstyle='->', 
#                        arrowsize=25, width=2)

# nx.draw_networkx_edges(G, pos, edgelist=non_linear_edges, 
#                        edge_color="blue", style="dashed", arrows=True, arrowstyle='->', 
#                        arrowsize=25, width=2)

# nx.draw_networkx_edges(G, pos, edgelist=linear_self_loop_edges, 
#                        edge_color="black", arrows=True, width=2,
#                        arrowsize=25)

# nx.draw_networkx_edges(G, pos, edgelist=non_linear_self_loop_edges, 
#                        edge_color="blue", style="dashed", arrows=False, 
#                        width=2)

# edges_AE = [(u, v) for u, v in G.edges() if {u, v} == {'A', 'E'} or {u, v} == {'E', 'A'}]
# nx.draw_networkx_edges(G, pos, edgelist=edges_AE, 
#                        edge_color="blue", style="dashed", arrows=True, 
#                        width=2, connectionstyle="arc3,rad=0.3")


# nx.draw_networkx_labels(G, pos, font_size=15, font_family="sans-serif")
# plt.axis('off') 
# plt.show()

