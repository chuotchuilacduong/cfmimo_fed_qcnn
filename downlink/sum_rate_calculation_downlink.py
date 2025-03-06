import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Environment.Environment import Environment
import numpy as np

# Random pilot
np.random.seed(0)

def random_pilot_assignment(env):
    tau = env.tau
    U = env.U
    K = env.K
    
    Phii = np.zeros((tau, K))
    for k in range(K):
        Point = np.random.randint(0, tau)
        Phii[:, k] = U[:, Point]
    return Phii

def calculate_gamma(M, K, tau, Pp, BETAA, Phii_cf):
    Gammaa = np.zeros((M, K))
    mau = np.zeros((M, K))
    
    for m in range(M):
        for k in range(K):
            mau[m, k] = np.linalg.norm((BETAA[m, :] ** 0.5) * (Phii_cf[:, k].T @ Phii_cf)) ** 2
    
    for m in range(M):
        for k in range(K):
            Gammaa[m, k] = tau * Pp * BETAA[m, k] ** 2 / (tau * Pp * mau[m, k] + 1)
    
    return Gammaa

def calculate_rate(env, BETAA, Gammaa, Phii_cf, Pu, Pd):
    M, K = BETAA.shape
    SINR = np.zeros(K)
    R_cf = np.zeros(K)
    
    # Equal power allocation for each AP
    etaa = np.zeros(M)
    for m in range(M):
        etaa[m] = 1 / np.sum(Gammaa[m, :])
    
    # Pilot contamination
    PC = np.zeros((K, K))
    for ii in range(K):
        for k in range(K):
            PC[ii, k] = np.sum((etaa ** 0.5) * ((Gammaa[:, ii] / BETAA[:, ii]) * BETAA[:, k])) * (Phii_cf[:, ii].T @ Phii_cf[:, k])
    PC1 = np.abs(PC) ** 2
    
    for k in range(K):
        num = 0
        for m in range(M):
            num += (etaa[m] ** 0.5) * Gammaa[m, k]
        SINR[k] = Pd * num ** 2 / (1 + Pd * np.sum(BETAA[:, k]) + Pd * np.sum(PC1[:, k]) - Pd * PC1[k, k])
        # Rate:
        R_cf[k] = np.log2(1 + SINR[k])
    
    return R_cf

