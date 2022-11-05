# 各種計算選擇權方式
def CRR_Binomial_Tree(S0, r, q, sigma, T, K, n, Type):
    """ 傳入參數後，用二項樹模型算出Call與Put的價格先回傳Call，在回傳Put
    ，可以用Type(European, American) """
    import math
    import numpy as np
    delta_t = T / n
    # 算出 u, d, p
    u = math.exp(sigma * math.sqrt(delta_t))
    d = 1.0 / u
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
            # 根據不同type去計算option price
            if Type.upper() == "EUROPEAN":
                c_opt_val[i, j] = math.exp(-r * delta_t) * (p *
                                                            c_opt_val[i + 1, j] + (1 - p) * c_opt_val[i + 1, j + 1])
                p_opt_val[i, j] = math.exp(-r * delta_t) * (p *
                                                            p_opt_val[i + 1, j] + (1 - p) * p_opt_val[i + 1, j + 1])
            if Type.upper() == "AMERICAN":
                c_opt_val[i, j] = max(
                    math.exp(-r * delta_t) * (p * c_opt_val[i + 1, j] + (1 - p) * c_opt_val[i + 1, j + 1]), st_tree[i, j] - K)
                p_opt_val[i, j] = max(
                    math.exp(-r * delta_t) * (p * p_opt_val[i + 1, j] + (1 - p) * p_opt_val[i + 1, j + 1]), K - st_tree[i, j])
    # 兩者都回傳時，先回傳put在回傳call
    return c_opt_val[0, 0], p_opt_val[0, 0]


def BlackSholes_model(S0, r, q, sigma, T, K):
    """ 傳入BS模型參數，用BS公式算出call、put價格後依序回傳 """
    from scipy.stats import norm
    import math
    d1 = (math.log(S0 / K) + (r - q + (sigma ** 2) * 0.5) * T) / \
        (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    N_negd1 = norm.cdf(-d1)
    N_negd2 = norm.cdf(-d2)
    # call price
    call = S0 * math.exp(-q * T) * Nd1 - K * math.exp(-r * T) * Nd2
    # put price
    put = K * math.exp(-r * T) * N_negd2 - S0 * math.exp(-q * T) * N_negd1
    return call, put


def CRR_Binomial_Tree_one_col(S0, r, q, sigma, T, K, n):
    """ 用一維陣列做CRR，只能用European，依序回傳call、put價格 """
    import math
    delta_t = T / n
    # 算出 u, d, p
    u = math.exp(sigma * math.sqrt(delta_t))
    d = 1.0 / u
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
    return c_opt_val[0], p_opt_val[0]


def CRR_Binomial_Tree_comb(S0, r, q, sigma, T, K, n):
    """ 利用組合數學的方式算出European的價格 """
    import math
    from scipy.special import comb
    delta_t = T / n
    # 算出 u, d, p
    u = math.exp(sigma * math.sqrt(delta_t))
    d = 1.0 / u
    p = (math.exp((r - q) * delta_t) - d) / (u - d)
    c_sum = 0
    p_sum = 0
    for j in range(n + 1):
        c_sum += comb(n, j) * p ** (n - j) * (1 - p) ** j * \
            max(S0 * u ** (n - j) * d ** j - K, 0)
        p_sum += comb(n, j) * p ** (n - j) * (1 - p) ** j * \
            max(K - (S0 * u ** (n - j) * d ** j), 0)
    # 折現
    c_opt_val = math.exp(-r * T) * c_sum
    p_opt_val = math.exp(-r * T) * p_sum
    return c_opt_val, p_opt_val


def print_resault(c, p, Type, Method):
    """ 傳入Call、Put價格以及是European或American還有是用什麼方法評價(Method, Bs or 各種CRR)
    ，並輸出結果 """
    print("=============================")
    print("%s:" % Method)
    print("%s call price: %0.5f" % (Type.title(), c))
    print("%s put price: %0.5f" % (Type.title(), p))
    print("=============================")
    print()


# 輸入
print()
# S0 = float(input("S0:"))
# r = float(input("r:"))
# q = float(input("q:"))
# sigma = float(input("sigma:"))
# T = float(input("T:"))
# K = float(input("K:"))
# n = int(input("n:"))  # n期
S0 = 35
K = 30
T = 2
sigma = 0.25
r = 0.05
q = 0.01
n = 500
print()
# 使用BS公式計算(只能是歐式)
c, p = BlackSholes_model(S0, r, q, sigma, T, K)
print_resault(c, p, Type="european", Method="Black-Sholes model")

# 算歐式選擇權(CRR)
c, p = CRR_Binomial_Tree(S0, r, q, sigma, T, K, n, Type="european")
print_resault(c, p, Type="european", Method="CRR binomial tree")

# 算美式選擇權(CRR)
c, p = CRR_Binomial_Tree(S0, r, q, sigma, T, K, n, Type="american")
print_resault(c, p, Type="american", Method="CRR binomial tree")

# 用一維的CRR算歐式選擇權
c, p = CRR_Binomial_Tree_one_col(S0, r, q, sigma, T, K, n)
print_resault(c, p, Type="european", Method="CRR binomial tree(one-column)")

# 用怎何樹學的方是計算歐式選擇權
c, p = CRR_Binomial_Tree_comb(S0, r, q, sigma, T, K, n)
print_resault(c, p, Type="european", Method="CRR binomial tree(combinatorial)")
