# 用二維陣列的CRR binomial--European
import math
import numpy as np
S0 = float(input("S0:"))
r = float(input("r:"))
q = float(input("q:"))
sigma = float(input("sigma:"))
T = float(input("T:"))
K = float(input("K:"))
n = int(input("n:"))  # n期
delta_t = T / n
# 算出 u, d, p
u = math.exp(sigma * math.sqrt(delta_t))
d = 1 / u
p = (math.exp((r - q) * delta_t) - d) / (u - d)
# 建立st tree
st_tree = np.zeros((n + 1, n + 1))
for i in range(0, n + 1):
    for j in range(0, i + 1):
        st_tree[i, j] = S0 * (u ** (i - j)) * (d ** j)
# optionvalue的矩陣
c_opt_val = np.zeros((n + 1, n + 1))
p_opt_val = np.zeros((n + 1, n + 1))
# 判斷最後那期的 option value
for j in range(n + 1):
    c_opt_val[n, j] = max(0, st_tree[n, j] - K)
    p_opt_val[n, j] = max(0, K - st_tree[n, j])
# backward induction
for i in range(n - 1, -1, -1):
    for j in range(i + 1):
        c_opt_val[i, j] = math.exp(-r * delta_t) * (p *
                                                    c_opt_val[i + 1, j] + (1 - p) * c_opt_val[i + 1, j + 1])
        p_opt_val[i, j] = math.exp(-r * delta_t) * (p *
                                                    p_opt_val[i + 1, j] + (1 - p) * p_opt_val[i + 1, j + 1])
print("=========================")
print("CRR binomial tree:")
print("European call price: %0.5f" % c_opt_val[0, 0])
print("European put price: %0.5f" % p_opt_val[0, 0])
print("=========================")
