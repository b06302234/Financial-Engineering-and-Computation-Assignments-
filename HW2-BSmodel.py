from scipy.stats import norm
import math
# 輸入
S0 = float(input("S0:"))
r = float(input("r:"))
q = float(input("q:"))
sigma = float(input("sigma:"))
T = float(input("T:"))
K = float(input("K:"))
d1 = (math.log(S0 / K) + (r - q + (sigma ** 2) * 0.5) * T) / \
    (sigma * math.sqrt(T))
d2 = d1 - sigma * math.sqrt(T)
Nd1 = norm.cdf(d1)
Nd2 = norm.cdf(d2)
N_negd1 = norm.cdf(-d1)
N_negd2 = norm.cdf(-d2)
# call price
c = S0 * math.exp(-q * T) * Nd1 - K * math.exp(-r * T) * Nd2
# put price
p = K * math.exp(-r * T) * N_negd2 - S0 * math.exp(-q * T) * N_negd1
print("=========================")
print("Black-Scholes model:")
print("call price: %0.5f" % c)
print("put price: %0.5f" % p)
print("=========================")
