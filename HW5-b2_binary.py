import math
import numpy as np
import time

# 輸入
print()
# S_t = float(input("S_t:"))  # initial stock price
# K = float(input("K:"))
# r = float(input("r:"))
# q = float(input("q:"))
# sigma = float(input("sigma:"))
# t = float(input("t:"))  # passing time
# T = float(input("T:"))  # maturality
# M = int(input("M:"))
# n = int(input("n:"))    # n-period tree
# Save_t = float(input("Save_t:"))
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
delta_t = (T - t) / n
tStart = time.time()
# calculate u, d, p
u = math.exp(sigma * math.sqrt(delta_t))
d = 1 / u
p = (math.exp((r - q) * delta_t) - d) / (u - d)
# binary search


def binary_search(arr, x):
    """ input an array and targrt x, return ku or kd """
    low = 0
    high = len(arr) - 1
    mid = 0
    while low <= high:
        mid = (high + low) // 2
        # Check if x is present at mid
        if arr[mid] < x:
            high = mid - 1
            if arr[mid] < x < arr[mid - 1]:
                return mid
        elif arr[mid] > x:
            low = mid + 1
            if arr[mid + 1] < x < arr[mid]:
                return mid + 1
        else:
            return mid
        # If we reach here, then the element was not present
    return -1


# average list
A = []
for i in range(n + 1):
    A.append([])
    for j in range(n + 1):
        A[i].append([])

# call
call = []
for i in range(n + 1):
    call.append([])
    for j in range(n + 1):
        call[i].append([])

# average list for each node
for i in range(n + 1):
    for j in range(i + 1):
        Amax = (Save_t * (t / delta_t + 1) + (S_t * u * (1 - u ** (i - j)) / (1 - u))
                + (S_t * (u ** (i - j)) * d * ((1 - (d ** j)) / (1 - d)))) / (i + 1 + (t / delta_t))
        Amin = (Save_t * (t / delta_t + 1) + (S_t * d * (1 - (d ** j)) / (1 - d))
                + (S_t * (d ** j) * u * (1 - u ** (i - j)) / (1 - u))) / (i + 1 + (t / delta_t))
        for k in range(M + 1):
            # linearly equally spaced
            Aijk = (M - k) / M * Amax + (k / M) * Amin
            A[i][j].append(Aijk)
# terminal payoff
for j in range(n + 1):
    for k in range(M + 1):
        payoff_k = max(A[n][j][k] - K, 0)
        call[n][j].append(payoff_k)

# backward induction
for i in range(n - 1, -1, -1):
    for j in range(i + 1):
        for k in range(M + 1):
            Au = ((i + 1 + t / delta_t) * A[i][j][k] + S_t *
                  (u ** (i + 1 - j)) * (d ** j)) / (i + 2 + (t / delta_t))
            Ad = ((i + 1 + t / delta_t) * A[i][j][k] + S_t * (u **
                                                              (i + 1 - (j + 1))) * (d ** (j + 1))) / (i + 2 + (t / delta_t))
            # binary search
            if A[i + 1][j][0] <= Au:
                cu = call[i + 1][j][0]
            elif A[i + 1][j][M] >= Au:
                cu = call[i + 1][j][M]
            else:
                ku = binary_search(A[i + 1][j], Au)
                wu = (A[i + 1][j][ku - 1] - Au) / \
                    (A[i + 1][j][ku - 1] - A[i + 1][j][ku])
                cu = wu * call[i + 1][j][ku] + \
                    (1 - wu) * call[i + 1][j][ku - 1]

            if A[i + 1][j + 1][0] <= Ad:
                cd = call[i + 1][j + 1][0]
            elif A[i + 1][j + 1][M] >= Ad:
                cd = call[i + 1][j + 1][M]
            else:
                kd = binary_search(A[i + 1][j + 1], Ad)
                wd = (A[i + 1][j + 1][kd - 1] - Ad) / \
                    (A[i + 1][j + 1][kd - 1] - A[i + 1][j + 1][kd])
                cd = wd * call[i + 1][j + 1][kd] + \
                    (1 - wd) * call[i + 1][j + 1][kd - 1]

            c = (p * cu + (1 - p) * cd) * math.exp(-r * delta_t)
            # early exercise
            if c < (A[i][j][k] - K):
                c = A[i][j][k] - K
            call[i][j].append(c)
tEnd = time.time()
print()
print("=============================")
print("CRR binomial tree for arithmetic call (binary search):")
print("option price: %0.5f" % call[0][0][0])
print()
print("It costs %0.2f sec" % (tEnd - tStart))
print("=============================")
print()
