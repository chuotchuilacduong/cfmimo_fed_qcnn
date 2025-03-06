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

def calculate_rate(env, BETAA, Gammaa, Phii_cf, Pu):
    M, K = BETAA.shape
    SINR = np.zeros(K)
    R_cf = np.zeros(K)
    
    # Pilot contamination
    PC = np.zeros((K, K))
    for ii in range(K):
        for k in range(K):
            PC[ii, k] = np.sum((Gammaa[:, k] / BETAA[:, k]) * BETAA[:, ii]) * (Phii_cf[:, k].T @ Phii_cf[:, ii])
    PC1 = np.abs(PC) ** 2
    
    for k in range(K):
        deno1 = 0
        for m in range(M):
            deno1 += Gammaa[m, k] * np.sum(BETAA[m, :])
        SINR[k] = Pu * (np.sum(Gammaa[:, k])) ** 2 / (np.sum(Gammaa[:, k]) + Pu * deno1 + Pu * np.sum(PC1[:, k]) - Pu * PC1[k, k])
        # Rate:
        R_cf[k] = np.log2(1 + SINR[k])
    
    return R_cf

if __name__ == "__main__":
    env = Environment()
    Phii_cf = random_pilot_assignment(env)
    BETAA = env.compute_large_scale_fading()
    Gammaa = calculate_gamma(env.M, env.K, env.tau, env.Pp, BETAA, Phii_cf)
    R_cf = calculate_rate(env, BETAA, Gammaa, Phii_cf, env.Pu)
    
    sum_rate = np.sum(R_cf)
    print("Rate per user (R_cf):")
    print(R_cf)
    print("Sum rate:")
    print(sum_rate)