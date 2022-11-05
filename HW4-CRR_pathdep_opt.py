import math
import numpy as np
import time

# 輸入
print()
S_t = float(input("S_t:"))  # 初始股價
r = float(input("r:"))
q = float(input("q:"))
sigma = float(input("sigma:"))
t = float(input("t:"))
T = float(input("T:"))
Smax_t = float(input("Smax_t:"))
n = int(input("n:"))    # n期的tree
delta_t = (T - t) / n
tStart = time.time()  # 計時開始
# 算出 u, d, p
u = math.exp(sigma * math.sqrt(delta_t))
d = 1 / u
p = (math.exp((r - q) * delta_t) - d) / (u - d)

# 建立st tree
st_tree = np.zeros((n + 1, n + 1))
# 先建立後兩期
for i in range(n - 1, n + 1):
    for j in range(0, i + 1):
        if (n % 2) != 0:
            if (i == (n - 1)) and (j == ((n - 1) / 2)):
                st_tree[i, j] = S_t
            else:
                st_tree[i, j] = S_t * (u ** (i - j)) * (d ** j)
        else:
            if (i == n) and (j == (n / 2)):
                st_tree[i, j] = S_t
            else:
                st_tree[i, j] = S_t * (u ** (i - j)) * (d ** j)
# 往前對應layer
for i in range(n - 2, -1, -1):
    for j in range(0, i + 1):
        st_tree[i, j] = st_tree[i + 2][j + 1]
# smax
smax = []
for i in range(n + 1):
    smax.append([])
    for j in range(n + 1):
        smax[i].append([])
# put
put = []
for i in range(n + 1):
    put.append([])
    for j in range(n + 1):
        put[i].append([])
# 建立smax tree(繼承)
smax[0][0].append(Smax_t)
for i in range(1, n + 1):
    for j in range(i + 1):
        if j == 0:
            for k in range(len(smax[i - 1][j])):
                if (smax[i - 1][j][k] >= st_tree[i][j]) and (smax[i - 1][j][k] not in smax[i][j]):
                    smax[i][j].append(smax[i - 1][j][k])
                elif (smax[i - 1][j][k] < st_tree[i][j]) and (smax[i - 1][j][k] not in smax[i][j]):
                    smax[i][j].append(st_tree[i][j])
        elif j == i:
            for k in range(len(smax[i - 1][j - 1])):
                if (smax[i - 1][j - 1][k] >= st_tree[i][j]) and (smax[i - 1][j - 1][k] not in smax[i][j]):
                    smax[i][j].append(smax[i - 1][j - 1][k])
                elif (smax[i - 1][j - 1][k] < st_tree[i][j]) and (smax[i - 1][j - 1][k] not in smax[i][j]):
                    smax[i][j].append(st_tree[i][j])
        else:
            for k in range(len(smax[i - 1][j - 1])):
                if (smax[i - 1][j - 1][k] >= st_tree[i][j]) and (smax[i - 1][j - 1][k] not in smax[i][j]):
                    smax[i][j].append(smax[i - 1][j - 1][k])
                elif (smax[i - 1][j - 1][k] < st_tree[i][j]) and (smax[i - 1][j - 1][k] not in smax[i][j]):
                    smax[i][j].append(st_tree[i][j])
            for k in range(len(smax[i - 1][j])):
                if (smax[i - 1][j][k] >= st_tree[i][j]) and (smax[i - 1][j][k] not in smax[i][j]):
                    smax[i][j].append(smax[i - 1][j][k])
                elif (smax[i - 1][j][k] < st_tree[i][j]) and (smax[i - 1][j][k] not in smax[i][j]):
                    smax[i][j].append(st_tree[i][j])
# 去除重複 + 排序
for i in range(n + 1):
    for j in range(n + 1):
        smaxset = set(smax[i][j])
        smax[i][j] = sorted(smaxset)
# 決定最後一期payoff
for j in range(n + 1):
    for k in range(len(smax[n][j])):
        # 每個node每個smax之下的payoff
        payoffi = max(smax[n][j][k] - st_tree[n][j], 0)
        put[n][j].append(payoffi)
# backward induction
for i in range(n - 1, -1, -1):
    for j in range(n + 1):
        for k in range(len(smax[i][j])):
            # 往上找
            if smax[i][j][k] in smax[i + 1][j]:
                index1 = smax[i + 1][j].index(smax[i][j][k])
                put_up = put[i + 1][j][index1]
            else:
                index2 = smax[i + 1][j].index(st_tree[i + 1][j])
                put_up = put[i + 1][j][index2]
            # 往下找
            index3 = smax[i + 1][j + 1].index(smax[i][j][k])
            put_down = put[i + 1][j + 1][index3]
            put_val = math.exp(-r * delta_t) * \
                (p * put_up + (1 - p) * put_down)

            # early exercise(拿掉就變european)
            if max(smax[i][j][k] - st_tree[i][j], 0) > put_val:
                put_val = max(smax[i][j][k] - st_tree[i][j], 0)
            put[i][j].append(put_val)
tEnd = time.time()  # 計時結束
print()
print("=============================")
print("CRR binomial tree for lookback put:")
print("option price: %0.5f" % put[0][0][0])
print()
print("It costs %0.2f sec" % (tEnd - tStart))
print("=============================")
print()
