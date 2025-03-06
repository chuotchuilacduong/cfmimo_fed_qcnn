import numpy as np

class Environment:
    def __init__(self, M=100, K=40, D=1, tau=20, B=20, Hb=15, Hm=1.65, f=1900, power_f=0.1, d0=0.01, d1=0.05, N=200, sigma_shd=8, D_cor=0.1):
        self.M = M
        self.K = K
        self.D = D
        self.tau = tau
        # Compute U matrix for tau orthogonal sequences
        self.U, _, _ = np.linalg.svd(np.random.randn(tau, tau))

        self.B = B
        self.Hb = Hb
        self.Hm = Hm
        self.f = f
        self.power_f = power_f
        self.d0 = d0
        self.d1 = d1
        self.N = N
        self.sigma_shd = sigma_shd
        self.D_cor = D_cor

        self.L = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(Hb) - ((1.1 * np.log10(f) - 0.7) * Hm - (1.56 * np.log10(f) - 0.8))
        self.noise_p = 10 ** ((-203.975 + 10 * np.log10(20 * 10**6) + 9) / 10)
        self.Pu = power_f / self.noise_p
        self.Pp = self.Pu

        self.AP = np.zeros((M, 2, 9))
        self.AP[:, :, 0] = np.random.uniform(-self.D/2, self.D/2, (self.M, 2))
        self.Ter = np.zeros((K, 2, 9))
        self.Ter[:, :, 0] = np.random.uniform(-D/2, D/2, (K, 2))

    def compute_large_scale_fading(self):
        Dist_AP = np.zeros((self.M, self.M))
        Cor_AP = np.zeros((self.M, self.M))
        for m1 in range(self.M):
            for m2 in range(self.M):
                Dist_AP[m1, m2] = min([np.linalg.norm(self.AP[m1, :, i] - self.AP[m2, :, i]) for i in range(9)])
                Cor_AP[m1, m2] = np.exp(-np.log(2) * Dist_AP[m1, m2] / self.D_cor)
        
        # Ensure the correlation matrix is positive definite
        Cor_AP += np.eye(self.M) * 1e-10
        
        A1 = np.linalg.cholesky(Cor_AP)
        x1 = np.random.randn(self.M, 1)
        sh_AP = A1 @ x1
        for m in range(self.M):
            sh_AP[m] = (1 / np.sqrt(2)) * self.sigma_shd * sh_AP[m] / np.linalg.norm(A1[m, :])

        Dist_Ter = np.zeros((self.K, self.K))
        Cor_Ter = np.zeros((self.K, self.K))
        for k1 in range(self.K):
            for k2 in range(self.K):
                Dist_Ter[k1, k2] = min([np.linalg.norm(self.Ter[k1, :, i] - self.Ter[k2, :, i]) for i in range(9)])
                Cor_Ter[k1, k2] = np.exp(-np.log(2) * Dist_Ter[k1, k2] / self.D_cor)
        
        # Ensure the correlation matrix is positive definite
        Cor_Ter += np.eye(self.K) * 1e-10
        
        A2 = np.linalg.cholesky(Cor_Ter)
        x2 = np.random.randn(self.K, 1)
        sh_Ter = A2 @ x2
        for k in range(self.K):
            sh_Ter[k] = (1 / np.sqrt(2)) * self.sigma_shd * sh_Ter[k] / np.linalg.norm(A2[k, :])

        Z_shd = np.zeros((self.M, self.K))
        for m in range(self.M):
            for k in range(self.K):
                Z_shd[m, k] = sh_AP[m] + sh_Ter[k]

        BETAA = np.zeros((self.M, self.K))
        dist = np.zeros((self.M, self.K))
        for m in range(self.M):
            for k in range(self.K):
                dist[m, k] = min([np.linalg.norm(self.AP[m, :, i] - self.Ter[k, :, 0]) for i in range(9)])
                if dist[m, k] < self.d0:
                    betadB = -self.L - 35 * np.log10(self.d1) + 20 * np.log10(self.d1) - 20 * np.log10(self.d0)
                elif self.d0 <= dist[m, k] <= self.d1:
                    betadB = -self.L - 35 * np.log10(self.d1) + 20 * np.log10(self.d1) - 20 * np.log10(dist[m, k])
                else:
                    betadB = -self.L - 35 * np.log10(dist[m, k]) + Z_shd[m, k]
                BETAA[m, k] = 10 ** (betadB / 10)
        return BETAA


