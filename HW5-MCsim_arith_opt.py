import numpy as np
import time
import math

# 輸入
print()
# S_t = float(input("S_t:"))  # 初始股價
# K = float(input("K:"))
# r = float(input("r:"))
# q = float(input("q:"))
# sigma = float(input("sigma:"))
# t = float(input("t:"))  # passing time
# T = float(input("T:"))  # maturality
# M = int(input("M:"))
# n = int(input("n:"))    # n期的tree
# Save_t = float(input("Save_t:"))
# numofsim = int(input("number of simulations:"))  # 抽樣幾個 ex: 10000
# numofrep = int(input("number of repetitions:"))  # 做幾次 ex: 20
S_t = 50  # 初始股價
K = 50
r = 0.1
q = 0.05
sigma = 0.8
t = 0.25  # passing time
T = 0.5  # maturality
M = 100
n = 100    # n期的tree
Save_t = 50
numofsim = 10000
numofrep = 20
tStart = time.time()  # 計時開始
delta_t = float((T - t) / n)
opt_val = np.zeros(numofrep)  # 每次模擬算出來的option price（加總平均折現後）
for rep in range(numofrep):
    payoffs = []    # 每次模擬每條路徑算出來的payoffs
    Spaths = np.zeros((numofsim, n + 2))    # 每個路徑的股價
    Spaths[:, 1] = S_t        # 每條路徑的初始值是原始股價
    Spaths[:, 0] = Save_t
    for col in range(2, n + 2):
        e = np.random.randn(numofsim)       # 用這個較快
        lnst = e * sigma * np.sqrt(delta_t) + np.log(Spaths[:, col - 1]) + (r - q - (sigma ** 2) / 2) * \
            delta_t
        Spaths[:, col] = np.exp(lnst)
    for row in range(numofsim):
        Save_T = (Save_t * (t / delta_t + 1) +
                  np.sum(Spaths[row, 2:])) / (n + (t / delta_t + 1))
        payoffi = max(Save_T - K, 0)
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
print("95% CI for arithmetic call: ")
print("mean: %0.5f" % mean)
print("std dev: %0.5f" % stddev)
print("upper bound: %0.5f" % upr)
print("lower bound: %0.5f" % lwr)
print()
print("It costs %0.2f sec" % (tEnd - tStart))
print("=============================")
