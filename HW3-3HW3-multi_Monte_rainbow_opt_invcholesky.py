import numpy as np
import math
print()
K = int(input("K:"))
r = float(input("r:"))
T = float(input("T"))
numofsim = int(input("number of simulations:"))  # 抽樣幾個 ex: 10000
numofrep = int(input("number of repetitions:"))  # 做幾次 ex: 20
n = int(input("n assets:"))
S0str = input("S0 of each asset:")
S0 = [int(i) for i in S0str.split()]    # n個資產的初始價格
qstr = input("q of each asset:")
q = [float(i) for i in qstr.split()]
# sigmastr = input("sigmas:")
# sigma = [float(i) for i in sigmastr.split()]
# print("rho martix:")
# rho = np.zeros((n, n))   # rho martix
# for i in range(n):
#     rhostr = input()
#     rho_row = rhostr.split()
#     for j in range(n):
#         rho[i][j] = float(rho_row[j])
# varcov = np.zeros((n, n))   # variance and covariance martix
# for i in range(n):
#     for j in range(n):
#         varcov[i][j] = float(rho[i][j] * sigma[i] * sigma[j] * T)
# varcov = np.zeros((n, n))   # variance and covariance martix
# for i in range(n):
#     for j in range(n):
#         varcov[i][j] = float(rho[i][j] * sigma[i] * sigma[j] * T)
print("var & cov martix:")
varcov = np.zeros((n, n))   # variance and covariance martix
for i in range(n):
    varcovstr = input()
    varcov_row = varcovstr.split()
    for j in range(n):
        varcov[i][j] = float(varcov_row[j])


def cholesky_decomp(varcov):
    """ 對傳入的varcov martix做cholesky，得到矩陣A(C = ATranspose * A) """
    A = np.zeros((n, n))
    A[0][0] = math.sqrt(varcov[0][0])
    for j in range(1, n):
        A[0][j] = varcov[0][j] / A[0][0]
    for i in range(1, n - 1):   # i index 已經調整過
        sum_aki2 = 0.0
        for k in range(0, i):
            sum_aki2 += (A[k][i] ** 2)
        A[i][i] = math.sqrt(varcov[i][i] - sum_aki2)
        for j in range(i, n):
            sum_akiakj = 0.0
            for k in range(0, i):
                sum_akiakj += A[k][i] * A[k][j]
            A[i][j] = (1 / A[i][i]) * (varcov[i][j] - sum_akiakj)
    sum_akn2 = 0.0
    for k in range(0, n):
        sum_akn2 += (A[k][n - 1] ** 2)
    A[n - 1][n - 1] = math.sqrt(varcov[n - 1][n - 1] - sum_akn2)
    return A


opt_val = []
A = cholesky_decomp(varcov)
for time in range(numofrep):
    Z = np.zeros((numofsim, n))
    # antithetic
    for col in range(0, n):
        for row in range(0, int(numofsim / 2)):
            Z[row][col] = np.random.randn(1)
        for row in range(int(numofsim / 2), numofsim):
            Z[row][col] = -1 * (Z[row - int(numofsim / 2)][col])
    sd = []
    for col in range(n):
        sd.append(math.sqrt(np.cov(Z, rowvar=False)[col][col]))
    # moment match
    for col in range(0, n):
        for row in range(numofsim):
            Z[row][col] = Z[row][col] / sd[col]
    # inverse cholesky
    B = cholesky_decomp(np.cov(Z, rowvar=False))
    Binv = np.linalg.inv(B)
    Astar = Binv.dot(A)
    R = Z.dot(Astar)

    # 計算mu
    mu = []
    for i in range(n):
        mui = math.log(S0[i]) + (r - q[i]) * T - (varcov[i][i] / 2)
        mu.append(mui)
    ST = np.zeros((numofsim, n))
    for col in range(n):
        for row in range(numofsim):
            ST[row][col] = math.exp(mu[col] + R[row][col])

    # payoff
    payoff = []
    for row in range(numofsim):
        payoffi = max(max(ST[row]) - K, 0)
        payoff.append(payoffi)

    # option value
    opt_val.append(math.exp(-r * T) * np.mean(payoff))
mean = np.mean(opt_val)
stddev = np.std(opt_val)
lwr = mean - 2 * stddev
upr = mean + 2 * stddev
print()
print("=========================")
print("95% CI for for rainbow option: ")
print("mean: %0.5f" % mean)
print("stddev: %0.5f" % stddev)
print("upper bound: %0.5f" % upr)
print("lower bound: %0.5f" % lwr)
print("=========================")
print()
