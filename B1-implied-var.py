import numpy as np
import scipy.stats as si
import math
import time

# implied volitity using newton's method
# BS model


def newton_vol_call_div_BS(S0, K, T, C, r, q):
    """ implied volitity using newton's method for european calls by BS model """
    MAX_ITERATIONS = 1000
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = BS_call(S0, K, T, r, q, sigma)
        vega = BS_vega(S0, K, T, r, q, sigma)
        price = price
        diff = C - price  # our root
        if (abs(diff) < PRECISION):
            # implied volitity
            return sigma
        sigma = sigma + diff/vega  # f(x) / f'(x)

    # value wasn't found, return best guess so far
    return sigma


def newton_vol_put_div_BS(S0, K, T, P, r, q):
    """ implied volitity using newton's method for european puts by BS model """
    MAX_ITERATIONS = 1000
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = BS_put(S0, K, T, r, q, sigma)
        vega = BS_vega(S0, K, T, r, q, sigma)
        price = price
        diff = P - price  # our root
        if (abs(diff) < PRECISION):
            # implied volitity
            return sigma
        sigma = sigma + diff/vega  # f(x) / f'(x)

    # value wasn't found, return best guess so far
    return sigma


def BS_vega(S0, K, T, r, q, sigma):
    """ Vega by using BS model """
    d1 = (math.log(S0 / K) + (r - q + (sigma ** 2) * 0.5) * T) / \
        (sigma * math.sqrt(T))
    vega = S0 * np.exp(-q * T) * np.sqrt(T) * si.norm.pdf(d1)
    return vega


def BS_call(S0, K, T, r, q, sigma):
    """ BS model for calls """
    d1 = (math.log(S0 / K) + (r - q + (sigma ** 2) * 0.5) * T) / \
        (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    Nd1 = si.norm.cdf(d1)
    Nd2 = si.norm.cdf(d2)
    # call price
    call = S0 * math.exp(-q * T) * Nd1 - K * math.exp(-r * T) * Nd2
    return call


def BS_put(S0, K, T, r, q, sigma):
    """ BS model for puts """
    d1 = (math.log(S0 / K) + (r - q + (sigma ** 2) * 0.5) * T) / \
        (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N_negd1 = si.norm.cdf(-d1)
    N_negd2 = si.norm.cdf(-d2)
    # put price
    put = K * math.exp(-r * T) * N_negd2 - S0 * math.exp(-q * T) * N_negd1
    return put

# --------------------------------------------------------
# CRR Tree


def newton_vol_call_div_CRRtree(S0, K, T, C, r, q, n, Type):
    """ implied volitity using newton's method for european calls by CRR tree """
    MAX_ITERATIONS = 1000
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = CRR_Binomial_Tree_call(S0, r, q, sigma, T, K, n, Type)
        h = 0.0001
        vega = (CRR_Binomial_Tree_call(S0, r, q, sigma + h, T, K, n,
                                       Type) - CRR_Binomial_Tree_call(S0, r, q, sigma, T, K, n, Type)) / h
        price = price
        diff = C - price  # our root
        if (abs(diff) < PRECISION):
            # implied volitity
            return sigma
        sigma = sigma + diff/vega  # f(x) / f'(x)

    # value wasn't found, return best guess so far
    return sigma


def CRR_Binomial_Tree_call(S0, r, q, sigma, T, K, n, Type):
    """ 傳入參數後，用二項樹模型算出Call
    ，可以用Type(European, American) """
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
    # 判斷最後那期的 option value
    for j in range(n + 1):
        c_opt_val[n, j] = max(0, st_tree[n, j] - K)
    # backward induction
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            # 根據不同type去計算option price
            if Type.upper() == "EUROPEAN":
                c_opt_val[i, j] = math.exp(-r * delta_t) * (p *
                                                            c_opt_val[i + 1, j] + (1 - p) * c_opt_val[i + 1, j + 1])
            if Type.upper() == "AMERICAN":
                c_opt_val[i, j] = max(
                    math.exp(-r * delta_t) * (p * c_opt_val[i + 1, j] + (1 - p) * c_opt_val[i + 1, j + 1]), st_tree[i, j] - K)
    return c_opt_val[0, 0]


def newton_vol_put_div_CRRtree(S0, K, T, P, r, q, n, Type):
    """ implied volitity using newton's method for european calls by CRR tree """
    MAX_ITERATIONS = 1000
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        price = CRR_Binomial_Tree_put(S0, r, q, sigma, T, K, n, Type)
        h = 0.0001
        vega = (CRR_Binomial_Tree_put(S0, r, q, sigma + h, T, K, n,
                                      Type) - CRR_Binomial_Tree_put(S0, r, q, sigma, T, K, n, Type)) / h
        price = price
        diff = P - price  # our root
        if (abs(diff) < PRECISION):
            # implied volitity
            return sigma
        sigma = sigma + diff/vega  # f(x) / f'(x)
    # value wasn't found, return best guess so far
    return sigma


def CRR_Binomial_Tree_put(S0, r, q, sigma, T, K, n, Type):
    """ 傳入參數後，用二項樹模型算出Put的價格
    ，可以用Type(European, American) """
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
    p_opt_val = np.zeros((n + 1, n + 1))
    # 判斷最後那期的 option value
    for j in range(n + 1):
        p_opt_val[n, j] = max(0, K - st_tree[n, j])
    # backward induction
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            # 根據不同type去計算option price
            if Type.upper() == "EUROPEAN":
                p_opt_val[i, j] = math.exp(-r * delta_t) * (p *
                                                            p_opt_val[i + 1, j] + (1 - p) * p_opt_val[i + 1, j + 1])
            if Type.upper() == "AMERICAN":
                p_opt_val[i, j] = max(
                    math.exp(-r * delta_t) * (p * p_opt_val[i + 1, j] + (1 - p) * p_opt_val[i + 1, j + 1]), K - st_tree[i, j])
    # 兩者都回傳時，先回傳put在回傳call
    return p_opt_val[0, 0]

# --------------------------------------------------------
# implied volitity using Bisection method


def Bisection_vol_call_div_BS(S0, K, T, C, r, q):
    a = 0.0001
    b = 2
    xn = 0
    fa = BS_call(S0, K, T, r, q, a) - C
    fb = BS_call(S0, K, T, r, q, b) - C
    if fa * fb < 0:
        while abs(a - b) > 1.0e-5:
            xn = (a + b) / 2
            fa = BS_call(S0, K, T, r, q, a) - C
            fxn = BS_call(S0, K, T, r, q, xn) - C
            if fa * fxn < 0:
                a = a   # lower bound
                b = xn  # upper bound
            else:
                a = xn
                b = b
        return xn
    else:
        print("root is not in a, b")


def Bisection_vol_put_div_BS(S0, K, T, P, r, q):
    a = 0.0001
    b = 2
    xn = 0
    fa = BS_put(S0, K, T, r, q, a) - P
    fb = BS_put(S0, K, T, r, q, b) - P
    if fa * fb < 0:
        while abs(a - b) > 1.0e-5:
            xn = (a + b) / 2
            fa = BS_put(S0, K, T, r, q, a) - P
            fxn = BS_put(S0, K, T, r, q, xn) - P
            if fa * fxn < 0:
                a = a   # lower bound
                b = xn  # upper bound
            else:
                a = xn
                b = b
        return xn
    else:
        print("root is not in a, b")


def Bisection_vol_call_div_CRRtree(S0, K, T, C, r, q, n, Type):
    a = 0.001
    b = 2
    xn = 0
    fa = CRR_Binomial_Tree_call(S0, r, q, a, T, K, n, Type) - C
    fb = CRR_Binomial_Tree_call(S0, r, q, b, T, K, n, Type) - C
    if fa * fb < 0:
        while abs(a - b) > 1.0e-5:
            xn = (a + b) / 2
            fa = CRR_Binomial_Tree_call(S0, r, q, a, T, K, n, Type) - C
            fxn = CRR_Binomial_Tree_call(S0, r, q, xn, T, K, n, Type) - C
            if fa * fxn < 0:
                a = a   # lower bound
                b = xn  # upper bound
            else:
                a = xn
                b = b
        return xn
    else:
        print("root is not in a, b")


def Bisection_vol_put_div_CRRtree(S0, K, T, P, r, q, n, Type):
    a = 0.001
    b = 2
    xn = 0
    fa = CRR_Binomial_Tree_put(S0, r, q, a, T, K, n, Type) - P
    fb = CRR_Binomial_Tree_put(S0, r, q, b, T, K, n, Type) - P
    if fa * fb < 0:
        while abs(a - b) > 1.0e-5:
            xn = (a + b) / 2
            fa = CRR_Binomial_Tree_put(S0, r, q, a, T, K, n, Type) - P
            fxn = CRR_Binomial_Tree_put(S0, r, q, xn, T, K, n, Type) - P
            if fa * fxn < 0:
                a = a   # lower bound
                b = xn  # upper bound
            else:
                a = xn
                b = b
        return xn
    else:
        print("root is not in a, b")


# results
# S0, K, T, P(C), r, q, n, Type
S0 = 50
K = 55
T = 0.5
P = 6.5
C = 2.5
r = 0.1
q = 0.03
n = 100
tStart = time.time()
print()
print("=============================")
print("Newton's Method:")
print("implied volitity (New + BS + call): %0.4f" %
      newton_vol_call_div_BS(S0, K, T, C, r, q))
print("implied volitity (New + BS + put): %0.4f" %
      newton_vol_put_div_BS(S0, K, T, P, r, q))
print("implied volitity (New + CRRtree + EUcall): %0.4f" % newton_vol_call_div_CRRtree(
    S0, K, T, C, r, q, n, "european"))
print("implied volitity (New + CRRtree + UScall): %0.4f" % newton_vol_call_div_CRRtree(
    S0, K, T, C, r, q, n, "american"))
print("implied volitity (New + CRRtree + EUput): %0.4f" % newton_vol_put_div_CRRtree(
    S0, K, T, P, r, q, n, "european"))
print("implied volitity (New + CRRtree + USput): %0.4f" % newton_vol_put_div_CRRtree(
    S0, K, T, P, r, q, n, "american"))
# --------------------------------------------------------
print()
print("Bisection:")
print("implied volitity (Bisection + BS + EUcall): %0.4f" %
      Bisection_vol_call_div_BS(S0, K, T, C, r, q))
print("implied volitity (Bisection + BS + EUput): %0.4f" %
      Bisection_vol_put_div_BS(S0, K, T, P, r, q))
print("implied volitity (Bisection  + CRRtree + EUcall): %0.4f" % Bisection_vol_call_div_CRRtree(
    S0, K, T, C, r, q, n, "european"))
print("implied volitity (Bisection  + CRRtree + UScall): %0.4f" % Bisection_vol_call_div_CRRtree(
    S0, K, T, C, r, q, n, "american"))
print("implied volitity (Bisection  + CRRtree + EUput): %0.4f" % Bisection_vol_put_div_CRRtree(
    S0, K, T, P, r, q, n, "european"))
print("implied volitity (Bisection  + CRRtree + USput): %0.4f" % Bisection_vol_put_div_CRRtree(
    S0, K, T, P, r, q, n, "american"))
tEnd = time.time()
print()
print("It costs %0.2f sec" % (tEnd - tStart))
print("=============================")
print()
