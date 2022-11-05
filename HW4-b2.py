import math
import numpy as np
import time

# 輸入
print()
# S_t = float(input("S_t:"))  # 初始股價
# r = float(input("r:"))
# q = float(input("q:"))
# sigma = float(input("sigma:"))
# t = float(input("t:"))
# T = float(input("T:"))
# Smax_t = float(input("Smax_t:"))
# n = int(input("n:"))    # n期的tree
S_t = 50
r = 0.1
q = 0
sigma = 0.4
t = 0
T = 0.25
Smax_t = 50
n = 1000
delta_t = (T - t) / n
tStart = time.time()  # 計時開始
# 算出 u, d, q
u = math.exp(sigma * math.sqrt(delta_t))
d = 1 / u
mu = math.exp((r - q) * delta_t)
qq = (mu * u - 1) / (mu * (u - d))
# Tree
st_tree = np.zeros((n + 1, n + 1))
for i in range(n + 1):
    for j in range(i + 1):
        st_tree[i, j] = u ** (i - j)
# payoff
payoff = np.zeros((n + 1, n + 1))

# terminal payoff
for j in range(n + 1):
    payoff[n, j] = max(st_tree[n, j] - 1, 0)

# backward induction
for i in range(n - 1, -1, -1):
    for j in range(i + 1):
        # if lowest layer
        if j == i:
            payoff[i, j] = mu * math.exp(-r * delta_t) * (qq * payoff[i + 1, j +
                                                                      1] + (1 - qq) * payoff[i + 1, j])
            # early exercise
            if payoff[i, j] < max(st_tree[i, j] - 1, 0):
                payoff[i, j] = max(st_tree[i, j] - 1, 0)
        else:
            payoff[i, j] = mu * math.exp(-r * delta_t) * (qq * payoff[i + 1, j +
                                                                      2] + (1 - qq) * payoff[i + 1, j])
            # early exercise
            if payoff[i, j] < max(st_tree[i, j] - 1, 0):
                payoff[i, j] = max(st_tree[i, j] - 1, 0)
print()
print("=============================")
print("put price: %0.4f" % (payoff[0, 0] * S_t))
print("=============================")
print()
