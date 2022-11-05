import numpy as np
import time
import math

# 輸入
print()
S_t = float(input("S_t:"))  # 初始股價
r = float(input("r:"))
q = float(input("q:"))
sigma = float(input("sigma:"))
t = float(input("t:"))
T = float(input("T:"))
Smax_t = float(input("Smax_t:"))
n = int(input("n:"))    # 切n期
numofsim = int(input("number of simulations:"))  # 抽樣幾個 ex: 10000
numofrep = int(input("number of repetitions:"))  # 做幾次 ex: 20
tStart = time.time()  # 計時開始
delta_t = float((T - t) / n)
opt_val = np.zeros(numofrep)  # 每次模擬算出來的option price（加總平均折現後）
for rep in range(numofrep):
    payoffs = []    # 每次模擬每條路徑算出來的payoffs
    Spaths = np.zeros((numofsim, n + 2))    # 每個路徑的股價
    Spaths[:, 1] = S_t        # 每條路徑的初始值是原始股價
    Spaths[:, 0] = Smax_t
    for col in range(2, n + 2):
        e = np.random.randn(numofsim)       # 用這個較快
        lnst = e * sigma * np.sqrt(delta_t) + np.log(Spaths[:, col - 1]) + (r - q - (sigma ** 2) / 2) * \
            delta_t
        Spaths[:, col] = np.exp(lnst)
    for row in range(numofsim):
        payoffi = max(max(Spaths[row]) -
                      Spaths[row, n], 0)
        payoffs.append(payoffi)
    avgpayoff = math.exp(-r * n * delta_t) * np.mean(payoffs)
    opt_val[rep] = avgpayoff  # 加總平均折現
mean = np.mean(opt_val)
stddev = np.std(opt_val)
upr = mean + 2 * stddev
lwr = mean - 2 * stddev
tEnd = time.time()  # 計時結束
print()
print("=============================")
print("Montecarlo simulation:")
print("95% CI for lookback put: ")
print("mean: %0.5f" % mean)
print("std dev: %0.5f" % stddev)
print("upper bound: %0.5f" % upr)
print("lower bound: %0.5f" % lwr)
print()
print("It costs %0.2f sec" % (tEnd - tStart))
print("=============================")
