import math
import statistics
import numpy as np
# 輸入
print()
S0 = float(input("S0:"))
r = float(input("r:"))
q = float(input("q:"))
sigma = float(input("sigma:"))
T = float(input("T:"))
K = float(input("K:"))
c_exp_20payoff = []  # for call
p_exp_20payoff = []  # for put
for i in range(20):
    c_payoff = []  # 抽樣10000的payoff for call
    p_payoff = []  # 抽樣10000的payoff for put
    for j in range(10000):
        e = np.random.randn(1)
        # variance轉成sigma * T
        # mean轉成 lns + (r - q - (sigma ** 2) / 2) * T)
        lnst = e * sigma * math.sqrt(T) + math.log(S0) + \
            (r - q - (sigma ** 2) / 2) * T
        # 再轉成股價
        st = math.exp(lnst)
        # 判斷payoff
        if st > K:
            c_payoff.append(st - K)
            p_payoff.append(0)
        else:
            c_payoff.append(0)
            p_payoff.append(K - st)
    c_avgpayoff = math.exp(-r * T) * statistics.mean(c_payoff)
    p_avgpayoff = math.exp(-r * T) * statistics.mean(p_payoff)
    c_exp_20payoff.append(c_avgpayoff)
    p_exp_20payoff.append(p_avgpayoff)
# call輸出
meanof20 = statistics.mean(c_exp_20payoff)
stddev = statistics.stdev(c_exp_20payoff)
upr = meanof20 + 2 * stddev
lwr = meanof20 - 2 * stddev
print()
print("=============================")
print("Montecarlo simulation:")
print("95% CI for call: ")
print("mean: %0.5f" % meanof20)
print("std dev: %0.5f" % stddev)
print("upper bound: %0.5f" % upr)
print("lower bound: %0.5f" % lwr)
print()
# put輸出
meanof20 = statistics.mean(p_exp_20payoff)
stddev = statistics.stdev(p_exp_20payoff)
upr = meanof20 + 2 * stddev
lwr = meanof20 - 2 * stddev
print("95% CI for put: ")
print("mean: %0.5f" % meanof20)
print("std dev: %0.5f" % stddev)
print("upper bound: %0.5f" % upr)
print("lower bound: %0.5f" % lwr)
print("=============================")
