import numpy as np
import math
import statistics

s0 = float(input("s0:"))
r = float(input("r:"))
q = float(input("q:"))
sigma = float(input("sigma:"))
T = float(input("T:"))
k1 = float(input("k1:"))
k2 = float(input("k2:"))
k3 = float(input("k3:"))
k4 = float(input("k4:"))
exp_20payoff = []
for i in range(20):
    payoff = []  # 抽樣10000的payoff
    for j in range(10000):
        e = np.random.randn(1)
        lnst = e * sigma * math.sqrt(T) + math.log(s0) + \
            (r - q - (sigma ** 2) / 2) * T
        st = math.exp(lnst)
        if k1 <= st <= k2:
            payoff.append(st - k1)
        elif k2 < st < k3:
            payoff.append(k2 - k1)
        elif k3 <= st <= k4:
            payoff.append((k2 - k1) / (k3 - k4) * st -
                          k4 * (k2 - k1) / (k3 - k4))
        else:
            payoff.append(0)
    avgpayoff = math.exp(-r * T) * statistics.mean(payoff)
    exp_20payoff.append(avgpayoff)
meanof20 = statistics.mean(exp_20payoff)
stddev = statistics.stdev(exp_20payoff)
upr = meanof20 + 2 * stddev
lwr = meanof20 - 2 * stddev
print("95% CI: ")
print("upper bound: %0.5f" % upr)
print("lower bound: %0.5f" % lwr)
