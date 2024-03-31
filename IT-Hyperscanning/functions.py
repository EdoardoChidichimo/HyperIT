#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: edoardochidichimo

METRICS:
    
    UNIVARIATE & JOINT SHANNON ENTROPY
    MUTUAL INFORMATION (& SYMMETRIC UNCERTAINTY)
    TRANSFER ENTROPY / CONDITIONAL MUTUAL INFORMATION
    GRANGER EMERGENCE


APPROACHES (Discrete [D] and Continuous [C] Estimators):

    [D] BINNING (Cohen inspired)
    [D] SYMBOLIC (Tyson Pond; rework)
    [C] GAUSSIAN COPULAS (Ince et al., 2017; verbatim)
    [C] KERNEL DENSITY 
    [C] k-NEAREST NEIGHBOUR
    

"""

# CORE
import io
from pathlib import Path
from copy import copy
from collections import OrderedDict
import requests

# DATA SCIENCE
import numpy as np
import scipy as sp
from scipy import stats
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.kde import KDEUnivariate
from statsmodels.nonparametric.kernel_density import KDEMultivariate

# HYPYP
from hypyp import prep 
from hypyp import analyses


# VISUALISATION
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
from mpl_toolkits.axes_grid1 import make_axes_locatable


# MNE
import mne
from mne.channels import make_standard_montage
from mne.viz import plot_topomap




# =============================================================================
# BINNING/HISTOGRAM APPROACH
# =============================================================================

def fd_bins_calc(X: np.array, Y: np.array):
     
    # Freedman-Diaconis Rule for Frequency-Distribution Bin Size
    fd_bins_X = np.ceil(np.ptp(X) / (2.0 * stats.iqr(X) * len(X)**(-1/3)))
    fd_bins_Y = np.ceil(np.ptp(Y) / (2.0 * stats.iqr(Y) * len(Y)**(-1/3)))
    fd_bins = int(np.ceil((fd_bins_X+fd_bins_Y)/2))
    
    return fd_bins
    
def pmd_hist(X: np.array, bins: int):
    '''
    UNIVARIATE PROBABILITY MASS DISTRIBUTION: BINNING/HISTOGRAM APPROACH
    
    Parameters
    ----------
    X : np.array
        Array of signal values.
    bins : int
        fd_bins should be passed here.

    Returns
    -------
    pmd_X : TYPE
        Probability Mass Distribution of random variable X.

    '''
    
    edges = np.linspace(min(X), max(X), bins)
    indices = np.digitize(X, edges, right=False)
    counts = np.bincount(indices, minlength=bins)[1:]
    pmd_X = counts.astype(float) / len(X)
    
    return pmd_X

def jpmd_hist(X: np.array, Y: np.array, bins: int, Z = None):
    '''
    JOINT PROBABILITY MASS DISTRIBUTION: BINNING/HISTOGRAM APPROACH
    
    Parameters
    ----------
    X : np.array
        Array of signal values.
    Y : np.array
        Array of signal values.
    bins : int
        fd_bins should be passed here.
    Z : TYPE, optional
        Array of signal values. The default is None.

    Returns
    -------
    TYPE
        If no Z, returns bivariate joint probability mass distribution, p_{XY}(X,Y)}.
        If Z is passed, returns trivariate joint probability mass distribution, p_{XYZ}(X,Y,Z)
    '''
    
    # Bin edges
    edges_X = np.linspace(min(X), max(X), bins)
    edges_Y = np.linspace(min(Y), max(Y), bins)
    
    # What bin each value should be in
    X_indices = np.digitize(X, edges_X, right=False)
    Y_indices = np.digitize(Y, edges_Y, right=False)
    
    # For each value, an array of bin length, with 1 indicating that bin (and 0 not)
    mask_X = (X_indices[:, None] == np.arange(1, bins + 1)).astype(int)
    mask_Y = (Y_indices[:, None] == np.arange(1, bins + 1)).astype(int)

    if Z is not None:
        
        assert Z.shape == X.shape == Y.shape, "All arrays should have the same shape"

        """
        freq, _ = np.histogramdd((X, Y, Z), bins=bins)
        test_pmd = freq / np.sum(freq)
        
        Why does this produce almost identical results (test_pmf - comb_pmf = 0 but np.array_equal(comb_pmf, test_pmf) = False)
        Go for numpy
        """
        
        edges_Z = np.linspace(min(Z), max(Z), bins)
        Z_indices = np.digitize(Z, edges_Z, right=False)
        
        comb_freq = np.zeros((bins, bins, bins))

        for i in range(bins):
            for j in range(bins):
                for k in range(bins):
                    XYZ_in_bin = (X_indices == i + 1) & (Y_indices == j + 1) & (Z_indices == k + 1)
                    comb_freq[i, j, k] = np.sum(XYZ_in_bin)

        comb_pmd = comb_freq / np.sum(comb_freq)
        
        return np.array(comb_pmd)

    else:
        freq = np.dot(mask_X.T, mask_Y).astype(float)
        jpmd = freq / np.sum(freq)
        
        return np.array(jpmd)


def entropy_hist(data: np.array):
    '''
    UNIVARIATE & JOINT SHANNON ENTROPY: BINNING/HISTOGRAM APPROACH
    
    Parameters
    ----------
    data : np.array
        shape is (2, n_epochs, n_channels, n_times).

    Returns
    -------
    np.array([H(X), H(Y), H(X,Y)]) 
        Average Shannon entropy *per channel*, both univariate and joint, 
        3 columns, n_epo rows.
    '''
    
    assert data[0].shape[0] == data[1].shape[0], "Two data streams should have the same number of trials."

    X = data[0]
    Y = data[1]
    
    n_epo, n_ch, n_samples = X.shape
    avg_ch_entropies = []
    
    for ch_i in range(n_ch):
        
        avg_ch_entropies_X = 0
        avg_ch_entropies_Y = 0
        avg_ch_joint_entropy = 0
        
        for epo_j in range(n_epo):
            
            # Timeseries for current channel and epoch
            Xi, Yi = X[epo_j, ch_i, :], Y[epo_j, ch_i, :]

            fd_bins = fd_bins_calc(Xi, Yi)
    
            # Calculate Univariate and Joint Probability Mass Distributions
            pmd_X = pmd_hist(Xi, fd_bins)
            pmd_Y = pmd_hist(Yi, fd_bins)
            jpmd = jpmd_hist(Xi, Yi, fd_bins)

            # Calculate Univariate and Joint Shannon Entropy
            avg_ch_entropies_X += -np.sum(pmd_X * np.log2(pmd_X + np.finfo(float).eps))
            avg_ch_entropies_Y += -np.sum(pmd_Y * np.log2(pmd_Y + np.finfo(float).eps))
            avg_ch_joint_entropy += -np.sum(jpmd * np.log2(jpmd + np.finfo(float).eps))
            
        avg_ch_entropies_X /= n_epo
        avg_ch_entropies_Y /= n_epo
        avg_ch_joint_entropy /= n_epo
        
        avg_ch_entropies.append([avg_ch_entropies_X, avg_ch_entropies_Y, avg_ch_joint_entropy])
    
    return np.array(avg_ch_entropies)
                    
def mi_hist(data: np.array):
    '''
    MUTUAL INFORMATION: BINNING/HISTOGRAM APPROACH

    Parameters
    ----------
    data : np.array
        shape is (2, n_epochs, n_channels, n_times).

    Returns
    -------
    MI : np.array
        Average Mutual Information *per channel*
        3 columns, n_epo rows..
    '''
    
    assert data[0].shape[0] == data[1].shape[0], "Two data streams should have the same number of trials."
    
    entropies = entropy_hist(data)
    MI = entropies[:,0] + entropies[:,1] - entropies[:,2]
    return MI

def te_hist(data: np.array, l = 3):
    
    # This is based on https://github.com/tysonpond/symbolic-transfer-entropy/blob/master/symbolic_TE.py
    # ALSO CHECK https://github.com/mariogutierrezroig/smite/blob/master/smite/core.py
    
    
    
    X = data[0]
    Y = data[1]

    assert X.shape == Y.shape, "Both signals must have the same shape"
    
    n_epo, n_ch, _ = X.shape
    #n_samples = X.shape[2] - l  # when l=3, n_samples = 501-3 = 498
    
    Xi = X[0][0][:-l]
    Yi = Y[0][0][:-l]
    Zi = Y[0][0][l:]
    
    fd_bins_X = np.ceil(np.ptp(Xi) / (2.0 * stats.iqr(Xi) * len(Xi)**(-1/3)))
    fd_bins_Y = np.ceil(np.ptp(Yi) / (2.0 * stats.iqr(Yi) * len(Yi)**(-1/3)))
    fd_bins_Z = np.ceil(np.ptp(Zi) / (2.0 * stats.iqr(Zi) * len(Zi)**(-1/3)))
    fd_bins = int(np.ceil((fd_bins_X+fd_bins_Y+fd_bins_Z)/3))
    
    pmd_X = pmd_hist(Xi, fd_bins)
    pmd_Y = pmd_hist(Yi, fd_bins)
    pmd_Z = pmd_hist(Zi, bins=fd_bins)
    pmd_XY = jpmd_hist(Xi, Yi, fd_bins)
    pmd_XZ = jpmd_hist(Xi, Zi, fd_bins)
    pmd_YZ = jpmd_hist(Yi, Zi, fd_bins)
    pmd_XYZ = jpmd_hist(Xi, Yi, fd_bins, Z=Zi)
    
    p_xyz, _ = np.histogramdd((Xi, Yi, Zi), bins=fd_bins, density=False)

    p_xz = np.sum(p_xyz, axis=1)[:, np.newaxis, :]
    p_xz /= np.sum(p_xz)
    
    p_yz = np.sum(p_xyz, axis=0)[np.newaxis, :, :]
    p_yz /= np.sum(p_yz)
    
    p_z = np.sum(p_xyz, axis=(0, 1))[np.newaxis, np.newaxis, :]
    p_z /= np.sum(p_z)
    
    pmd_X_given_Z = pmd_XZ / (pmd_Z + np.finfo(float).eps)
    pmd_X_given_YZ = pmd_XYZ / (pmd_YZ + np.finfo(float).eps)
    
    cmi_1 = -np.sum(pmd_XYZ * np.log2((pmd_X_given_YZ / (pmd_X_given_Z + np.finfo(float).eps)) + np.finfo(float).eps))
    cmi_2 = -np.sum(pmd_XZ * np.log2(pmd_X_given_Z + np.finfo(float).eps) + pmd_XYZ * np.log2(pmd_X_given_YZ + np.finfo(float).eps))
    
    
    
    print(cmi_1)
    print(cmi_2)
    
    # P(A|B) = P(A;B) / P(B)
    
    #          - p(XYZ) * log(p(X|YZ) / p(X|Z))
    # I(X;Y) = H(H|Z) - H(X|Y,Z) 
    #        = − p(XZ)log(p(X∣Z)) + p(XYZ)log(p(X∣YZ))
    
    return None


# For TE/CMI using KL-divergence: https://github.com/notsebastiano/transfer_entropy/blob/master/TE.py







# =============================================================================
# SYMBOLIC APPROACH
# =============================================================================

def symbolise(X: np.ndarray, l=3, m=3):
    
    Y = np.empty((m, len(X) - (m - 1) * l))
    for i in range(m):
        Y[i] = X[i * l:i * l + Y.shape[1]]
    return Y.T
        
def incr_counts(key,d):
    d[key] = d.get(key, 0) + 1

def normalise(d):
    s=sum(d.values())        
    for key in d:
        d[key] /= s


def entropy_symb(epo1: mne.Epochs, epo2: mne.Epochs, l=3, m=3):
    
    a = epo1.get_data(copy=False)
    b = epo2.get_data(copy=False)
    
    n_epo, n_ch, n_samples = a.shape
    avg_ch_entropies = []
    
    hashmult = np.power(m, np.arange(m))
        
    
    for ch_i in range(n_ch):
        
        avg_ch_entropies_X = 0
        avg_ch_entropies_Y = 0
        avg_ch_entropies_XY = 0
    
        for epo_j in range(n_epo):
        
            X, Y = a[epo_j, ch_i, :], b[epo_j, ch_i, :]
    
            X = symbolise(X, l, m).argsort(kind='quicksort')
            Y = symbolise(Y, l, m).argsort(kind='quicksort')
    
            hashval_X = (np.multiply(X, hashmult)).sum(1) # multiply each symbol [1,0,3] by hashmult [1,3,9] => [1,0,27] and give a final array of the sum of each code ([.., .., 28, .. ])
            hashval_Y = (np.multiply(Y, hashmult)).sum(1)
            
            x_sym_to_perm = hashval_X
            y_sym_to_perm = hashval_Y
            
            p_xy = {}
            p_x = {}
            p_y = {}
            
            for i in range(len(x_sym_to_perm)-1):
                xy = str(x_sym_to_perm[i]) + "," + str(y_sym_to_perm[i])
                x = str(x_sym_to_perm[i])
                y = str(y_sym_to_perm[i])
                
                incr_counts(xy,p_xy)
                incr_counts(x,p_x)
                incr_counts(y,p_y)
                
            normalise(p_xy)
            normalise(p_x)
            normalise(p_y)
            
            p_xy = np.array(list(p_xy.values()))
            p_x = np.array(list(p_x.values()))
            p_y = np.array(list(p_y.values()))
            
            avg_ch_entropies_XY += -np.sum(p_xy * np.log2(p_xy + np.finfo(float).eps))
            avg_ch_entropies_X += -np.sum(p_x * np.log2(p_x + np.finfo(float).eps))
            avg_ch_entropies_Y += -np.sum(p_y * np.log2(p_y + np.finfo(float).eps))
    
        avg_ch_entropies_X /= n_epo
        avg_ch_entropies_Y /= n_epo
        avg_ch_entropies_XY /= n_epo
        
        avg_ch_entropies.append([avg_ch_entropies_X, avg_ch_entropies_Y, avg_ch_entropies_XY])
    
    return np.array(avg_ch_entropies)

def mi_symb(epo1: mne.Epochs, epo2: mne.Epochs, l=3, m=3):
    
    entropies = entropy_symb(epo1, epo2, l, m)
    MI = entropies[:,0] + entropies[:,1] - entropies[:,2]
    return MI

def te_symb(epo1: mne.Epochs, epo2: mne.Epochs, l=3, m=3):

    
    X = symbolise(epo1.get_data(copy=False)[0][0][:], l, m).argsort(kind='quicksort')
    Y = symbolise(epo2.get_data(copy=False)[0][0][:], l, m).argsort(kind='quicksort')    
    
    hashmult = np.power(m, np.arange(m))
    hashval_X = (np.multiply(X, hashmult)).sum(1) # multiply each symbol [1,0,3] by hashmult [1,3,9] => [1,0,27] and give a final array of the sum of each code ([.., .., 28, .. ])
    hashval_Y = (np.multiply(Y, hashmult)).sum(1)
    
    x_sym_to_perm = hashval_X
    y_sym_to_perm = hashval_Y #len = 495
    
    p_xyz = {}
    p_xy = {}
    p_yz = {}
    p_y = {}
    
    for i in range(len(y_sym_to_perm)-1):
        xyz = str(x_sym_to_perm[i]) + "," + str(y_sym_to_perm[i]) + "," + str(y_sym_to_perm[i+1])
        xy = str(x_sym_to_perm[i]) + "," + str(y_sym_to_perm[i])
        yz = str(y_sym_to_perm[i]) + "," + str(y_sym_to_perm[i+1])
        y = str(y_sym_to_perm[i])
        #z = str(z_sym_to_perm[i])

        incr_counts(xyz,p_xyz)
        incr_counts(xy,p_xy)
        incr_counts(yz,p_yz)
        incr_counts(y,p_y)
        
    normalise(p_xyz)
    normalise(p_xy)
    normalise(p_yz)
    normalise(p_y)
    
    p_z_given_xy = p_xyz.copy()
    for key in p_z_given_xy:
        xy_symb = key.split(",")[0] + "," + key.split(",")[1]
        p_z_given_xy[key] /= p_xy[xy_symb]    
    # also works: p_z_given_xy = {xyz: p_xyz[xyz] / p_xy[xyz.split(",")[0] + "," + xyz.split(",")[1]] for xyz in p_xyz}  
    
    p_z_given_y = p_yz.copy()
    for key in p_z_given_y:
        y_symb = key.split(",")[0]
        p_z_given_y[key] /= p_y[y_symb]
        
    
    # p_x_given_yz = p_xyz.copy()
    # for key in p_x_given_yz:
    #     yz_symb = key.split(",")[1] + "," + key.split(",")[2]
    #     p_x_given_yz[key] /= p_yz[yz_symb]
        
    # p_x_given_z = 

    final_sum = 0
    for key in p_xyz:
        yz_symb = key.split(",")[1] + "," +  key.split(",")[2] 
        if key in p_z_given_xy and yz_symb in p_z_given_y:
            if float('-inf') < float(np.log2(p_z_given_xy[key]/p_z_given_y[yz_symb])) < float('inf'):
                final_sum += p_xyz[key]*np.log2(p_z_given_xy[key]/p_z_given_y[yz_symb])
    
    return final_sum








# =============================================================================
# GAUSSIAN COPULA APPROACH (INCE)
# =============================================================================

# Gaussian Copula has been copied verbatim from Ince!

def copnorm(X: np.array):
    
    # Compute Empirical CDF
    xi = np.argsort(X)
    xr = np.argsort(xi) 
    """ Why ARGsort twice? (Argsort finds INDICES whilst sort actually takes the values of the array and moves them) """
    ecdf = (xr+1).astype(float) / (xr.shape[-1]+1)
    
    # Compute the inverse CDF (aka percent-point function or quantile function or inverse normal distribution)
    cx = sp.special.ndtri(ecdf) #around xbar = 0, sd = 1 (since it is normal!)
    return cx

def ent_g(X: np.array, biascorrect=True):
    """Entropy of a Gaussian variable in bits

    H = ent_g(x) returns the entropy of a (possibly 
    multidimensional) Gaussian variable x with bias correction.
    Columns of x correspond to samples, rows to dimensions/variables. 
    (Samples last axis)
    
    BASED ON INCE'S GCMI TOOLBOX

    """
    X = np.atleast_2d(X)
    
    Ntrl = X.shape[1]
    Nvarx = X.shape[0]

    # demean data
    X = X - X.mean(axis=1)[:,np.newaxis]
    # covariance
    C = np.dot(X,X.T) / float(Ntrl - 1)
    chC = np.linalg.cholesky(C)

    # entropy in nats
    HX = np.sum(np.log(np.diagonal(chC))) + 0.5*Nvarx*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi((Ntrl - np.arange(1,Nvarx+1).astype(float))/2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl-1.0)) / 2.0
        HX = HX - Nvarx*dterm - psiterms.sum()

    # convert to bits
    return HX / ln2
    
    
def mi_gc(data: np.array, biascorrect=True):
    
    """ Gaussian-Copula Mutual Information between two continuous variables.
    
    MI = mi_gc(epo1,epo2) returns the MI between two (possibly multidimensional)
    Gassian variables, x and y, with bias correction.
    
    If x and/or y are multivariate columns must correspond to samples, rows
    to dimensions/variables. (Samples last axis) 
    This provides a lower bound to the true MI value.
    
    Parameters
    ----------
    epo1 : mne.Epochs
        Signal 1.
    epo2 : mne.Epochs
        Signal 2.
    biascorrect : boolean, optional
        DESCRIPTION. The default is True. Specifies whether
        bias correction should be applied to the esimtated MI.

    Returns
    -------
    MI : TYPE
        DESCRIPTION.

    """
    
    X = data[0] 
    Y = data[1]
    
    x = copnorm(np.atleast_2d(X)) # instead of shape (501,) --> (1, 501), x.ndim = 2
    y = copnorm(np.atleast_2d(Y))
    
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarxy = Nvarx+Nvary

    if y.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # joint variable
    xy = np.vstack((x,y))
    
    Cxy = np.dot(xy,xy.T) / float(Ntrl - 1)
    # submatrices of joint covariance
    Cx = Cxy[:Nvarx,:Nvarx]
    Cy = Cxy[Nvarx:,Nvarx:]

    chCxy = np.linalg.cholesky(Cxy)
    chCx = np.linalg.cholesky(Cx)
    chCy = np.linalg.cholesky(Cy)

    # entropies in nats
    # normalizations cancel for mutual information
    HX = np.sum(np.log(np.diagonal(chCx))) # + 0.5*Nvarx*(np.log(2*np.pi)+1.0)
    HY = np.sum(np.log(np.diagonal(chCy))) # + 0.5*Nvary*(np.log(2*np.pi)+1.0)
    HXY = np.sum(np.log(np.diagonal(chCxy))) # + 0.5*Nvarxy*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi((Ntrl - np.arange(1,Nvarxy+1)).astype(float)/2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl-1.0)) / 2.0
        HX = HX - Nvarx*dterm - psiterms[:Nvarx].sum()
        HY = HY - Nvary*dterm - psiterms[:Nvary].sum()
        HXY = HXY - Nvarxy*dterm - psiterms[:Nvarxy].sum()

    # MI in bits
    MI = (HX + HY - HXY) / ln2
    return MI

def cmi_gc(X: np.array, Y: np.array, Z: np.array, biascorrect=True):
    
    x = copnorm(np.atleast_2d(X)) # instead of shape (501,) --> (1, 501), x.ndim = 2
    y = copnorm(np.atleast_2d(Y))
    z = copnorm(np.atleast_2d(Z))
    
    Ntrl = x.shape[1]
    Nvarx = x.shape[0]
    Nvary = y.shape[0]
    Nvarz = z.shape[0]
    Nvaryz = Nvary + Nvarz
    Nvarxy = Nvarx + Nvary
    Nvarxz = Nvarx + Nvarz
    Nvarxyz = Nvarx + Nvaryz

    if y.shape[1] != Ntrl or z.shape[1] != Ntrl:
        raise ValueError("number of trials do not match")

    # joint variable
    xyz = np.vstack((x,y,z))
    
    Cxyz = np.dot(xyz,xyz.T) / float(Ntrl - 1)
    # submatrices of joint covariance
    Cz = Cxyz[Nvarxy:,Nvarxy:]
    Cyz = Cxyz[Nvarx:,Nvarx:]
    Cxz = np.zeros((Nvarxz,Nvarxz))
    Cxz[:Nvarx,:Nvarx] = Cxyz[:Nvarx,:Nvarx]
    Cxz[:Nvarx,Nvarx:] = Cxyz[:Nvarx,Nvarxy:]
    Cxz[Nvarx:,:Nvarx] = Cxyz[Nvarxy:,:Nvarx]
    Cxz[Nvarx:,Nvarx:] = Cxyz[Nvarxy:,Nvarxy:]

    chCz = np.linalg.cholesky(Cz)
    chCxz = np.linalg.cholesky(Cxz)
    chCyz = np.linalg.cholesky(Cyz)
    chCxyz = np.linalg.cholesky(Cxyz)

    # entropies in nats
    # normalizations cancel for cmi
    HZ = np.sum(np.log(np.diagonal(chCz))) # + 0.5*Nvarz*(np.log(2*np.pi)+1.0)
    HXZ = np.sum(np.log(np.diagonal(chCxz))) # + 0.5*Nvarxz*(np.log(2*np.pi)+1.0)
    HYZ = np.sum(np.log(np.diagonal(chCyz))) # + 0.5*Nvaryz*(np.log(2*np.pi)+1.0)
    HXYZ = np.sum(np.log(np.diagonal(chCxyz))) # + 0.5*Nvarxyz*(np.log(2*np.pi)+1.0)

    ln2 = np.log(2)
    if biascorrect:
        psiterms = sp.special.psi((Ntrl - np.arange(1,Nvarxyz+1)).astype(float)/2.0) / 2.0
        dterm = (ln2 - np.log(Ntrl-1.0)) / 2.0
        HZ = HZ - Nvarz*dterm - psiterms[:Nvarz].sum()
        HXZ = HXZ - Nvarxz*dterm - psiterms[:Nvarxz].sum()
        HYZ = HYZ - Nvaryz*dterm - psiterms[:Nvaryz].sum()
        HXYZ = HXYZ - Nvarxyz*dterm - psiterms[:Nvarxyz].sum()

    # MI in bits
    CMI = (HXZ + HYZ - HXYZ - HZ) / ln2
    return CMI









# =============================================================================
# k-NEAREST NEIGHBOUR: MUTUAL INFORMATION, CONDITIONAL MUTUAL INFORMATION
# =============================================================================

## PYCOPENT (Ma Jian) -- uses copula entropy and transfer entropy using kNN method (Chebyshev distance)
## NPEET (https://github.com/gregversteeg/NPEET/blob/master/npeet/entropy_estimators.py)

# First to propose: Kraskov et al., 2004 (aka KSG method)


def mi_knn():
    # check pycopent, npeet.entropy_estimators
    return None

def cmi_knn():
    # check https://github.com/omesner/knncmi/blob/master/knncmi/knncmi.py
    return None







# =============================================================================
# KERNAL-DENSITY ESTIMATION APPROACH:
# =============================================================================

# See https://github.com/pablo-crucera/MI-estimation/blob/main/r-scripts/main.R
# First to propose: Moon et al., 1995

# def epanechnikov_kernel(u):
#     return 0.75 * (1 - u**2) 

def optimal_h(d, n):
    return (4/(d+2))**(1/(d+4)) * n**(-1/(d+4))

def entropy_kde(epo1: mne.Epochs, epo2: mne.Epochs, h: float | None = None, kernel: str = 'epa'):
    
    X = epo1.get_data(copy=False)[0][0]
    Y = epo2.get_data(copy=False)[0][0]
    
    # Find optimal "smoothing parameter" (or "bandwidth" or "h"), same for X and Y
    h = optimal_h(1,len(X)) * 1e-5
    """h ~ .33, but far too insensitive, .33e-5 works. (Is this because it is in the range of EEG values?)"""
    
    # # Joint KDE (assuming independence to obtain maximal entropy)
    # pdf_XY = pdf_X * pdf_Y
    # pdf_XY /= np.sum(pdf_XY)
    
    
    #POINTWISE (evaluate at each X value) 
    X_kde = KDEUnivariate(X).fit()
    X_pdf = X_kde.evaluate(X)
    X_pdf /= np.sum(X_pdf)
    ent_X = -np.sum(X_pdf * np.log2(X_pdf + np.finfo(float).eps))
    
    #1000 SUPPORT (does this mean as SUPPORT --> inf, the estimation becomes more accurate?)
    support = np.linspace(min(X)-h,max(X)+h, 1000)
    X_gpdf = X_kde.evaluate(support)
    X_gpdf /= np.sum(X_gpdf)
    X_ent = -np.sum(X_gpdf * np.log2(X_gpdf + np.finfo(float).eps))
    
    # Variation of Entropy values with different support sizes
    entropy_values = []
    support_sizes = np.logspace(1, 6, num=100, base=10) 
    
    for size in support_sizes:
        support = np.linspace(min(X) - X_kde.bw, max(X) + X_kde.bw, int(size))
        X_gpdf = X_kde.evaluate(support)
        X_gpdf /= np.sum(X_gpdf)
        X_ent = -np.sum(X_gpdf * np.log2(X_gpdf + np.finfo(float).eps))
        entropy_values.append(X_ent)
    
    plt.plot(support_sizes, entropy_values)
    plt.title('Variation of Entropy with Support Size')
    plt.xlabel('Support Size')
    plt.ylabel('Entropy')
    plt.show()
    
    
    
    
    
    
    kde_X = KernelDensity(bandwidth=h, kernel=kernel).fit(X.reshape(-1,1))
    kde_Y = KernelDensity(bandwidth=h, kernel=kernel).fit(Y.reshape(-1,1))
    
    # POINTWISE (for global, replace X.reshape... with linspace of 1000s of points)
    pdf_X = np.exp(kde_X.score_samples(X.reshape(-1,1)))
    pdf_Y = np.exp(kde_Y.score_samples(Y.reshape(-1,1)))
    
    # Normalise
    pdf_X /= np.sum(pdf_X)
    pdf_Y /= np.sum(pdf_Y)
    
    # GLOBAL 
    """Based on the following, how should we compute the pdf? Pointwise or extrapolate/global?"""
    # Plotting entropy against support sizes
    entropy_values = []
    support_sizes = np.logspace(1, 5, num=100, base=10) 
    
    for size in support_sizes:
        support = np.linspace(min(X) - h, max(X) + h, int(size))
        X_gpdf = np.exp(kde_X.score_samples(support.reshape(-1,1)))
        X_gpdf /= np.sum(X_gpdf)
        X_ent = -np.sum(X_gpdf * np.log2(X_gpdf + np.finfo(float).eps))
        entropy_values.append(X_ent)
    
    plt.plot(support_sizes, entropy_values)
    plt.title('Variation of Entropy with Support Size')
    plt.xlabel('Support Size')
    plt.ylabel('Entropy')
    plt.show()
    
    
    
    # POINTWISE: Joint KDE using KDEMultivariate
    joint_kde = KDEMultivariate((X,Y), bw=(h,h), var_type='cc')
    joint_pdf = joint_kde.pdf(np.column_stack([X,Y]))
    joint_pdf /= np.sum(joint_pdf)
    """Should we take maximal entropy and assume independence ==> p(x,y)=p(x)p(y)?"""
    
    
    # Calculate Entropies
    entropy_X = -np.sum(pdf_X * np.log2(pdf_X + np.finfo(float).eps))
    entropy_Y = -np.sum(pdf_Y * np.log2(pdf_Y + np.finfo(float).eps))
    joint_entropy = -np.sum(joint_pdf * np.log2(joint_pdf + np.finfo(float).eps))
    
    
    
    """Which:
                I(X;Y) = H(X)+H(Y)-H(X,Y) 
                      or 
                I(X;Y) = sumX sumY p(XY)log(p(XY)/p(X)p(Y))"""
    
    mut_info1 = entropy_X + entropy_Y - joint_entropy
    mut_info2 = np.sum(joint_pdf * np.log2((joint_pdf / (pdf_X * pdf_Y)) + np.finfo(float).eps))
    
    
    return np.array([entropy_X, entropy_Y, joint_entropy])
    
    """ Should the pdf be calculated pointwise, and then cycle through and calculate Shannon IC and averaged -> H(X)
        Or, should we take the global estimated pdf and calculate average entropy?
    """
    
    # =============================================================================
    #     VISUALISATION
    # =============================================================================
    
    X_plot = np.linspace(np.min(X)-h, np.max(X)+h, 1000)[:, np.newaxis]
    Y_plot = np.linspace(np.min(Y)-h, np.max(Y)+h, 1000)[:, np.newaxis]
    
    kde_X = KernelDensity(bandwidth=h, kernel=kernel).fit(X.reshape(-1,1))
    kde_Y = KernelDensity(bandwidth=h, kernel=kernel).fit(Y.reshape(-1,1))
    
    
    # Since .score_samples returns the log density, we need to exp()
    pdf_X = np.exp(kde_X.score_samples(X_plot))
    pdf_Y = np.exp(kde_Y.score_samples(Y_plot))
    
    # Normalise
    pdf_X /= np.sum(pdf_X)
    pdf_Y /= np.sum(pdf_Y)
    
    
    
    # FIGURE 1 — Signals, Prob Distr, Univariate and Joint KDE

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 9)) 
    
    # EEG Value against Time for X
    axes[0, 0].plot(X)  
    axes[0, 0].set_title("EEG Signal (X)")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Amplitude (μV)")
    
    # EEG Value against Time for Y
    axes[0, 1].plot(Y)
    axes[0, 1].set_title("EEG Signal (Y)")
    axes[0, 1].set_xlabel("Time (ms)")
    axes[0, 1].set_ylabel("Amplitude (μV)")
    
    # Histograms with Twin Axes
    fd_bins = fd_bins_calc(X, Y)
    axes[1, 0].hist(X, bins=fd_bins, density=True, alpha=0.5, label='Histogram X')
    axes[1, 0].twinx().plot(X_plot, pdf_X, color='red', label='KDE X')
    axes[1, 0].legend()
    axes[1, 0].set_xlim(np.min(X) - h, np.max(X) + h)
    axes[1, 0].set_xlabel("Amplitude (μV)")
    
    axes[1, 1].hist(Y, bins=fd_bins, density=True, alpha=0.5, label='Histogram Y')
    axes[1, 1].twinx().plot(Y_plot, pdf_Y, color='red', label='KDE Y')
    axes[1, 1].legend()
    axes[1, 1].set_xlim(np.min(Y) - h, np.max(Y) + h)
    axes[1, 1].set_xlabel("Amplitude (μV)")
    
    # Scatter Plot
    axes[2, 0].scatter(np.vstack((X,Y))[0], np.vstack((X,Y))[1], alpha=0.5, s = 5)
    axes[2, 0].set_title("Scatterplot")
    axes[2, 0].set_xlabel("X")
    axes[2, 0].set_ylabel("Y")
    axes[2, 0].set_xlim(np.min(X) - h, np.max(X) + h)
    axes[2, 0].set_ylim(np.min(Y) - h, np.max(Y) + h)
    
    # Joint KDE Plot
    sns.kdeplot(x=np.vstack((X,Y))[0], y=np.vstack((X,Y))[1], fill=True, cmap="Blues", bw_method=optimal_h(1, len(X)), ax=axes[2, 1])
    axes[2, 1].set_title("Joint Distribution KDE")
    axes[2, 1].set_xlabel("X")
    axes[2, 1].set_ylabel("Y")
    axes[2, 1].set_xlim(np.min(X) - h, np.max(X) + h)
    axes[2, 1].set_ylim(np.min(Y) - h, np.max(Y) + h)
    
    plt.tight_layout()
    plt.show()

    
    # FIGURE 2 - Multivariate KDE 2D Scatterplot + Kernels

    xmin, xmax = min(X)-h, max(X)+h
    ymin, ymax = min(Y)-h, max(Y)+h
    xgrid, ygrid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    grid_points = np.column_stack([xgrid.ravel(), ygrid.ravel()])
    
    joint_pdf_vis = joint_kde.pdf(grid_points)
    density_estimation = joint_pdf_vis.reshape(100, 100)

    plt.scatter(X, Y, alpha=0.5, label='Data', s=5)
    plt.contour(xgrid, ygrid, density_estimation, cmap='Blues', levels=10, alpha=0.7)
    plt.xlabel('X Amplitude (µV)')
    plt.ylabel('Y Amplitude (µV)')
    plt.legend()
    plt.title('Multivariate KDE')
    plt.show()
    
    
    # FIGURE 3 - Multivariate KDE 3D Plot
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(xgrid, ygrid, density_estimation, cmap='viridis', alpha=1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('PDF Value')
    ax.set_title('Joint KDE (3D)')
    
    plt.show()

    
    # FIGURE 4 - Joint and Marginal Distributions with KDE Overlay 
    
    sns.set()
    grid = sns.JointGrid(x=X, y=Y, marginal_ticks=True)
    
    # Scatter plot
    sns.kdeplot(x=X, y=Y, fill=True, cmap="Blues", ax=grid.ax_joint)
    grid.plot_joint(sns.scatterplot, color='gray', s=5)
    
    # Histograms
    grid.ax_marg_x.hist(X, bins=fd_bins, color='blue', alpha=0.7)
    grid.ax_marg_x.twinx().plot(X_plot, pdf_X, color='red', label='KDE X')
    
    grid.ax_marg_y.hist(Y, bins=fd_bins, color='blue', alpha=0.7, orientation='horizontal')
    grid.ax_marg_y.twinx().plot(Y_plot, pdf_Y, color='red', label='KDE Y')
    
    grid.set_axis_labels('Signal X (μV)', 'Signal Y (μV)')
    grid.fig.suptitle('Joint and Marginal Distributions with KDE Overlay', y=1.02)
    plt.show()
     
def mi_kde(epo1: mne.Epochs, epo2: mne.Epochs, vis=False):
    
    entropies = entropy_kde(epo1, epo2, kernel='epanechnikov')
    MI = entropies[:,0] + entropies[:,1] - entropies[:,2]
    
    return MI
    
    
    











freq_bands = {'Alpha-Low': [7.5, 11],
              'Alpha-High': [11.5, 13]}
full_freq = { 'full_frq': [1, 48]}
freq_bands = OrderedDict(full_freq)


URL_TEMPLATE = "https://github.com/ppsp-team/HyPyP/blob/master/data/participant{}-epo.fif?raw=true"

def get_data(idx):
    return io.BytesIO(requests.get(URL_TEMPLATE.format(idx)).content)

epo1 = mne.read_epochs(get_data(1), preload=True) 
epo2 = mne.read_epochs(get_data(2), preload=True)
mne.epochs.equalize_epoch_counts([epo1, epo2])


sampling_rate = epo1.info['sfreq']


### PRE-PROCESSING EPOCHS
icas = prep.ICA_fit([epo1, epo2], n_components=15, method='infomax', fit_params=dict(extended=True), random_state = 42)

cleaned_epochs_ICA = prep.ICA_choice_comp(icas, [epo1, epo2])


#auto-reject
cleaned_epochs_AR, dic_AR = prep.AR_local(cleaned_epochs_ICA, strategy="union", threshold=50.0, verbose=True)

#picking the preprocessed epoch for each participant
preproc_S1 = cleaned_epochs_AR[0]
preproc_S2 = cleaned_epochs_AR[1]

# Combine preprocessed data 
data = np.array([preproc_S1, preproc_S2])







## PLOTTING FOR THESIS
test = data[0][0][0]


plt.plot(test)
plt.title('EEG Signal')
plt.xlabel('Time (ms)')
plt.ylabel('Voltage (μV)')
plt.show()


fft_result = np.fft.fft(test)
fft_freqs = np.fft.fftfreq(len(test), 1 / 100)
fft_magnitude = np.abs(fft_result**2) 

plt.figure(figsize=(10, 6))
plt.plot(fft_freqs[:len(fft_freqs)//2], fft_magnitude[:len(fft_magnitude)//2])
plt.title('Frequency-Power Plot')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power')
plt.grid(True)
plt.show()








## VISUALISE EEG SIGNALS

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(a[0][0], color='blue')
plt.title('EEG Signal X, Channel 1, Epoch 1')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (μV)')
plt.grid(True)
plt.subplot(1, 2, 2)
plt.plot(b[0][0], color='orange')
plt.title(f'EEG Signal Y, Channel 1, Epoch 1')    
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (μV)')
plt.grid(True)
plt.tight_layout()
plt.show()


## VISUALISE SHANNON ENTROPY

results = entropy_hist(data)
pp1, pp2 = results[:, 0], results[:, 1]

montage = make_standard_montage('standard_1020')
info1 = mne.create_info(epo1.info.ch_names, sfreq=1, ch_types='eeg').set_montage(montage)
info2 = mne.create_info(epo2.info.ch_names, sfreq=1, ch_types='eeg').set_montage(montage)

# Create an Evoked object with average entropy values
evoked1 = mne.EvokedArray(pp1[:, np.newaxis], info1)
evoked2 = mne.EvokedArray(pp2[:, np.newaxis], info2)


fig, axes = plt.subplots(1,2,figsize=(15,6))
plt.subplots_adjust(wspace=.001)

img1,_ = mne.viz.plot_topomap(results[:,0], evoked1.info, ch_type = 'eeg', cmap='viridis',
                     vlim=[np.min(results), np.max(results)], show=False, sensors=True, contours=0, axes=axes[0])
axes[0].set_xlabel('Participant 1', fontsize=15)

img2,_ = mne.viz.plot_topomap(results[:,1], evoked2.info, ch_type = 'eeg', cmap='viridis',
                     vlim=[np.min(results), np.max(results)], show=False, sensors=True, contours=0, axes=axes[1])
axes[1].set_xlabel('Participant 2', fontsize=15)

divider = make_axes_locatable(axes[1])
cax = divider.append_axes('right', size='10%', pad=.5)
cbar = plt.colorbar(img2, cax=cax, orientation='vertical')
cbar.set_label('Entropy in bits')

plt.suptitle('Topomaps of Average Shannon Entropy', fontsize=25)
plt.show()
















# mi_hist_value = mi_hist(epo1, epo2, vis=True) #MI values per channel!
# mi_symb(epo1, epo2, vis=True)
# for l in range(1,5):
#     for m in range(1,5):
#         print('l =',l,', m =',m)
#         print('TE:', te_symb(epo1, epo2, l, m))
                        
