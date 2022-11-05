import math
from scipy.special import comb
# 輸入
print()
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
d = 1.0 / u
p = (math.exp((r - q) * delta_t) - d) / (u - d)
c_sum = 0
p_sum = 0


def comb_k(n, j):
    sum1 = 0
    for i in range(1, n + 1):
        sum1 += math.log(i)
    sum1 = sum1 + ((n - j) * math.log(p) + (j) * math.log(1 - p))
    sum2 = 0
    for i in range(1, j + 1):
        sum2 += math.log(i)
    sum3 = 0
    for i in range(1, n - j + 1):
        sum3 += math.log(i)
    sum4 = sum1 - sum2 - sum3
    return math.exp(sum4)


for j in range(n + 1):
    c_sum += comb_k(n, j) * max(S0 * (u ** (n - j)) * (d ** j) - K, 0)
    p_sum += comb_k(n, j) * max(K - (S0 * (u ** (n - j)) * (d ** j)), 0)
c_opt_val = math.exp(-r * T) * c_sum
p_opt_val = math.exp(-r * T) * p_sum
print(c_opt_val, p_opt_val)
