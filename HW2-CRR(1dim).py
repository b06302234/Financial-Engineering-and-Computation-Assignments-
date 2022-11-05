import math
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
d = 1 / u
p = (math.exp((r - q) * delta_t) - d) / (u - d)
# 最後一期的st
st_final = []
for j in range(n + 1):
    st_final.append(S0 * (u ** (n - j)) * (d ** j))
# 最後一期的option value
c_opt_val = []
p_opt_val = []
for j in range(n + 1):
    c_opt_val.append(max(0, st_final[j] - K))
    p_opt_val.append(max(0, K - st_final[j]))
# backward incuction
for i in range(n - 1, -1, -1):
    for j in range(i + 1):
        c_opt_val[j] = math.exp(-r * delta_t) * \
            (p * c_opt_val[j] + (1 - p) * c_opt_val[j + 1])
        p_opt_val[j] = math.exp(-r * delta_t) * \
            (p * p_opt_val[j] + (1 - p) * p_opt_val[j + 1])
print()
print("============================")
print("CRR binomial tree(one col):")
print("European call price: %0.5f" % c_opt_val[0])
print("European put price: %0.5f" % p_opt_val[0])
print("============================")
print()
