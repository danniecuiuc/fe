# %%[markdown]
# # Problem Set 3

# %%
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import datetime as tm

# %%
start = 50
n = 1000
s0 = 100
k = 95
sigma = 0.3
r = 0.1
t = 0.2
q = 0

# %%[markdown]
# # Question 1: Plain Europen Put


# %%[markdown]
# ### B-S pricing
# %%
def black_scholes(c_p, s0, k, t, r, q, sigma):

    cdf_mean = 0.0
    cdf_sd = 1.0

    d1 = (np.log(s0 / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = (np.log(s0 / k) + (r - q - 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))

    if c_p == 'call':

        call = s0 * np.exp(-q * t) * st.norm.cdf(d1, cdf_mean, cdf_sd)
        call = call - k * np.exp(-r * t) * st.norm.cdf(d2, cdf_mean, cdf_sd)

        return call

    else:

        put = k * np.exp(-r * t) * st.norm.cdf(-d2, cdf_mean, cdf_sd)
        put = put - s0 * np.exp(-q * t) * st.norm.cdf(-d1, cdf_mean, cdf_sd)

        return put


# %%
bsp_value = black_scholes('put', s0, k, t, r, q, sigma)
bsp_value

# %%[markdown]
# ### (1) CRR Model


# %%
def crre(s0, k, t, r, sigma, start, n):

    crr_result = []

    option_value = np.zeros([n + 1, n + 1])

    stock_value = np.zeros([n + 1, n + 1])

    for n in range(start, n + 1, 1):

        dt = t / n

        u = np.exp(sigma * np.sqrt(dt))

        d = 1 / u

        qu = (np.exp(r * dt) - d) / (u - d)

        qd = 1 - qu

        j = n

        for i in range(j + 1):

            stock_value[j, i] = s0 * u**i * d**(j - i)
            option_value[j, i] = np.maximum(k - stock_value[j, i], 0)

        for j in range(n - 1, -1, -1):

            for i in range(j, -1, -1):

                pv = np.exp(-r * dt) * (qu * option_value[j + 1, i + 1] +
                                        qd * option_value[j + 1, i])
                option_value[j, i] = pv

        output = {'num_steps': n, 'CRRE': option_value[0, 0]}

        crr_result.append(output)

    return (crr_result)


# %%
result = crre(s0, k, t, r, sigma, start, n)
# %%
crre_result = pd.DataFrame.from_dict(result)
crre_result['error_CRRE'] = bsp_value - crre_result['CRRE']
print(crre_result.head())
print(crre_result.tail())
# %%
crre_result.to_csv('../data/hw3_1_crre.csv')

# %%
crre_result = pd.read_csv('../data/hw3_1_crre.csv')
plt.figure(figsize=(80 / 3, 15))
plt.plot(crre_result['num_steps'],
         crre_result['error_CRRE'],
         'o',
         markersize=5)
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Error of CRR Model for European Put')
plt.grid()
plt.show()

# %%
extrp_n = np.arange(50, 501, 50)
extrp_value = [
    (i * crre_result['CRRE'][i - 50] - 2 * i * crre_result['CRRE'][2 * i - 50])
    / (-i) for i in extrp_n
]
extrp_error = -np.array(extrp_value) + bsp_value

# %%
plt.figure(figsize=(80 / 3, 15))
plt.plot(extrp_n,
         extrp_error,
         '-',
         linewidth=10,
         label='Extrapulated Error Terms at N&M')
plt.plot(extrp_n,
         crre_result['error_CRRE'][2 * extrp_n - 50],
         linewidth=10,
         label='Unextrapulated Error Terms at M=2N')
plt.legend(prop={'size': 30})
plt.xlabel('Extrapolated N')
plt.ylabel('Error')
plt.title('Error of Extrapolated CRR Model for European Put')
plt.grid()
plt.show()


# %%[markdown]
# ### (2) Rendleman and Bartter Model
# %%
def rbe(s0, k, t, r, sigma, start, n):

    rb_result = []

    option_value = np.zeros([n + 1, n + 1])

    stock_value = np.zeros([n + 1, n + 1])

    for n in range(start, n + 1, 1):

        dt = t / n

        u = np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt))

        d = np.exp((r - q - 0.5 * sigma**2) * dt - sigma * np.sqrt(dt))

        qu = (np.exp(r * dt) - d) / (u - d)

        qd = 1 - qu

        j = n

        for i in range(j + 1):

            stock_value[j, i] = s0 * u**i * d**(j - i)
            option_value[j, i] = np.maximum(k - stock_value[j, i], 0)

        for j in range(n - 1, -1, -1):

            for i in range(j, -1, -1):

                pv = np.exp(-r * dt) * (qu * option_value[j + 1, i + 1] +
                                        qd * option_value[j + 1, i])
                option_value[j, i] = pv

        output = {'num_steps': n, 'RBE': option_value[0, 0]}

        rb_result.append(output)

    return (rb_result)


# %%
result = rbe(s0, k, t, r, sigma, start, n)
# %%
rbe_result = pd.DataFrame.from_dict(result)
rbe_result['error_RBE'] = bsp_value - rbe_result['RBE']
print(rbe_result.head())
print(rbe_result.tail())
# %%
rbe_result.to_csv('../data/hw3_1_rbe.csv')

# %%
rbe_result = pd.read_csv('../data/hw3_1_rbe.csv')
plt.figure(figsize=(80 / 3, 15))
plt.plot(rbe_result['num_steps'], rbe_result['error_RBE'], 'o', markersize=5)
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Error of Rendleman and Bartter Model for European Put')
plt.grid()
plt.show()

# %%
extrp_n = np.arange(50, 501, 50)
extrp_value = [
    (i * rbe_result['RBE'][i - 50] - 2 * i * rbe_result['RBE'][2 * i - 50]) /
    (-i) for i in extrp_n
]
extrp_error = -np.array(extrp_value) + bsp_value

# %%
plt.figure(figsize=(80 / 3, 15))
plt.plot(extrp_n,
         extrp_error,
         '-',
         linewidth=10,
         label='Extrapulated Error Terms at N&M')
plt.plot(extrp_n,
         rbe_result['error_RBE'][2 * extrp_n - 50],
         linewidth=10,
         label='Unextrapulated Error Terms at M=2N')
plt.legend(prop={'size': 30})
plt.xlabel('Extrapolated N')
plt.ylabel('Error')
plt.title(
    'Error of Extrapolated Rendleman and Bartter Model Model for European Put')
plt.grid()
plt.show()


# %%[markdown]
# ### (3) Leisen and Reimer Model
# %%
def lre(s0, k, t, r, sigma, start, n):

    lr_result = []

    option_value = np.zeros([n + 1, n + 1])

    stock_value = np.zeros([n + 1, n + 1])

    def h(x):

        h_result = 0.5 + np.sign(x) * np.sqrt(0.25 -
                                              0.25 * np.exp(-(n + 1 / 6) *
                                                            (x /
                                                             (n + 1 / 3))**2))

        return h_result

    for n in range(start, n + 1, 1):

        dt = t / n

        d1 = (np.log(s0 / k) +
              (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = (np.log(s0 / k) +
              (r - q - 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))

        Q = h(d2)
        u = (np.exp((r - q) * dt) * h(d1)) / Q
        d = (np.exp((r - q) * dt) - Q * u) / (1 - Q)

        qu = (np.exp(r * dt) - d) / (u - d)
        qd = 1 - qu

        j = n

        for i in range(j + 1):

            stock_value[j, i] = s0 * u**i * d**(j - i)
            option_value[j, i] = np.maximum(k - stock_value[j, i], 0)

        for j in range(n - 1, -1, -1):

            for i in range(j, -1, -1):

                pv = np.exp(-r * dt) * (qu * option_value[j + 1, i + 1] +
                                        qd * option_value[j + 1, i])
                option_value[j, i] = pv

        output = {'num_steps': n, 'LRE': option_value[0, 0]}

        lr_result.append(output)

    return (lr_result)


# %%
t_start = tm.datetime.now()
result = lre(s0, k, t, r, sigma, start, n)
t_end = tm.datetime.now()
print(str(t_end - t_start))

# %%
lre_result = pd.DataFrame.from_dict(result)
lre_result['error_LRE'] = bsp_value - lre_result['LRE']
print(lre_result.head())
print(lre_result.tail())
# %%
lre_result.to_csv('../data/hw3_1_lre.csv')

# %%
lre_result = pd.read_csv('../data/hw3_1_lre.csv')
plt.figure(figsize=(80 / 3, 15))
plt.plot(lre_result['num_steps'], lre_result['error_LRE'], 'o', markersize=5)
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Error of Leisen and Reimer Model for European Put')
plt.grid()
plt.show()

# %%
extrp_n = np.arange(51, 501, 50)
extrp_value = [
    (i**2 * lre_result['LRE'][i - 50] -
     (2 * i - 1)**2 * lre_result['LRE'][2 * i - 51]) / (i**2 - (2 * i - 1)**2)
    for i in extrp_n
]
extrp_error = -np.array(extrp_value) + bsp_value

# %%
plt.figure(figsize=(80 / 3, 15))
plt.plot(extrp_n,
         extrp_error,
         '-',
         linewidth=10,
         label='Extrapulated Error Terms at N&M')
plt.plot(extrp_n,
         lre_result['error_LRE'][2 * extrp_n - 51],
         linewidth=10,
         label='Unextrapulated Error Terms at M=2N-1')
plt.legend(prop={'size': 30})
plt.xlabel('Extrapolated N')
plt.ylabel('Error')
plt.title('Error of Extrapolated Leisen and Reimer Model for European Put')
plt.grid()
plt.show()

# %%[markdown]
# # Question 2: American Put


# %%[markdown]
# ### American Binomial pricing
# %%
def crra_10000(s0, k, t, r, sigma, start, n=10000):

    option_value = np.zeros([n + 1, n + 1])

    stock_value = np.zeros([n + 1, n + 1])

    dt = t / n

    u = np.exp(sigma * np.sqrt(dt))

    d = 1 / u

    qu = (np.exp(r * dt) - d) / (u - d)

    qd = 1 - qu

    j = n

    for i in range(j + 1):

        stock_value[j, i] = s0 * u**i * d**(j - i)
        option_value[j, i] = np.maximum(k - stock_value[j, i], 0)

    for j in range(n - 1, -1, -1):

        for i in range(j, -1, -1):

            stock_value[j, i] = s0 * u**i * d**(j - i)

            pv_hold = np.exp(-r * dt) * (qu * option_value[j + 1, i + 1] +
                                         qd * option_value[j + 1, i])
            pv_exercise = np.maximum(k - stock_value[j, i], 0)

            option_value[j, i] = np.maximum(pv_hold, pv_exercise)

    output = option_value[0, 0]

    return (output)


# %%
t_start = tm.datetime.now()
ame_value = crra_10000(s0, k, t, r, sigma, start, 10000)
print(ame_value)
t_end = tm.datetime.now()
print(str(t_end - t_start))

# %%
ame_value = 2.5189053957724434  # N=10000


# %%[markdown]
# ### (1) CRR Model
# %%
def crra(s0, k, t, r, sigma, start, n):

    crr_result = []

    option_value = np.zeros([n + 1, n + 1])

    stock_value = np.zeros([n + 1, n + 1])

    for n in range(start, n + 1, 1):

        dt = t / n

        u = np.exp(sigma * np.sqrt(dt))

        d = 1 / u

        qu = (np.exp(r * dt) - d) / (u - d)

        qd = 1 - qu

        j = n

        for i in range(j + 1):

            stock_value[j, i] = s0 * u**i * d**(j - i)
            option_value[j, i] = np.maximum(k - stock_value[j, i], 0)

        for j in range(n - 1, -1, -1):

            for i in range(j, -1, -1):

                stock_value[j, i] = s0 * u**i * d**(j - i)

                pv_hold = np.exp(-r * dt) * (qu * option_value[j + 1, i + 1] +
                                             qd * option_value[j + 1, i])
                pv_exercise = np.maximum(k - stock_value[j, i], 0)

                option_value[j, i] = np.maximum(pv_hold, pv_exercise)

        output = {'num_steps': n, 'CRRA': option_value[0, 0]}

        crr_result.append(output)

    return (crr_result)


# %%
t_start = tm.datetime.now()
result = crra(s0, k, t, r, sigma, start, n)
t_end = tm.datetime.now()
print(str(t_end - t_start))

# %%
crra_result = pd.DataFrame.from_dict(result)
crra_result['error_CRRA'] = ame_value - crra_result['CRRA']
print(crra_result.head())
print(crra_result.tail())
# %%
crra_result.to_csv('../data/hw3_2_crra.csv')

# %%
crra_result = pd.read_csv('../data/hw3_2_crra.csv')
plt.figure(figsize=(80 / 3, 15))
plt.plot(crra_result['num_steps'],
         crra_result['error_CRRA'],
         'o',
         markersize=5)
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Error of CRR Model for American Put')
plt.grid()
plt.show()

# %%
extrp_n = np.arange(50, 501, 50)
extrp_value = [
    (i * crra_result['CRRA'][i - 50] - 2 * i * crra_result['CRRA'][2 * i - 50])
    / (-i) for i in extrp_n
]
extrp_error = -np.array(extrp_value) + ame_value

# %%
plt.figure(figsize=(80 / 3, 15))
plt.plot(extrp_n,
         extrp_error,
         '-',
         linewidth=10,
         label='Extrapulated Error Terms at N&M')
plt.plot(extrp_n,
         crra_result['error_CRRA'][2 * extrp_n - 50],
         linewidth=10,
         label='Unextrapulated Error Terms at M=2N')
plt.legend(prop={'size': 30})
plt.xlabel('Extrapolated N')
plt.ylabel('Error')
plt.title('Error of Extrapolated CRR Model for American Put')
plt.grid()
plt.show()

# %%[markdown]
# ## CRRA exercise boundary with N=100

# %%
steps = np.arange(n - 1, -1, -1)
n = 100
j = n

boundary = [{'step': n, 'exercise price': k}]

option_value = np.zeros([n + 1, n + 1])
stock_value = np.zeros([n + 1, n + 1])
pv_exercise = np.zeros([n + 1, n + 1])

dt = t / n

u = np.exp(sigma * np.sqrt(dt))
d = 1 / u

qu = (np.exp(r * dt) - d) / (u - d)
qd = 1 - qu

for i in range(j + 1):

    stock_value[j, i] = s0 * u**i * d**(j - i)
    option_value[j, i] = np.maximum(k - stock_value[j, i], 0)

for j in steps:

    S = []

    for i in np.arange(j, -1, -1):

        stock_value[j, i] = s0 * u**i * d**(j - i)
        option_value[j,
                     i] = np.exp(-r * dt) * (qu * option_value[j + 1, i + 1] +
                                             qd * option_value[j + 1, i])
        pv_exercise[j, i] = np.maximum(k - stock_value[j, i], 0)

        if pv_exercise[j, i] > option_value[j, i]:

            S = S + [stock_value[j, i]]

    if S != []:

        exercise_price = max(list([x for x in S]))

        result = {'step': j, 'exercise price': exercise_price}

    boundary.append(result)

df = pd.DataFrame.from_dict(boundary)
plt.plot(df['step'], df['exercise price'])

# %%[markdown]
# ### (2) Leisen and Reimer Model


# %%
def lra(s0, k, t, r, sigma, start, n):

    lr_result = []

    option_value = np.zeros([n + 1, n + 1])

    stock_value = np.zeros([n + 1, n + 1])

    def h(x):

        h_result = 0.5 + np.sign(x) * np.sqrt(0.25 -
                                              0.25 * np.exp(-(n + 1 / 6) *
                                                            (x /
                                                             (n + 1 / 3))**2))

        return h_result

    for n in range(start, n + 1, 1):

        dt = t / n

        d1 = (np.log(s0 / k) +
              (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
        d2 = (np.log(s0 / k) +
              (r - q - 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))

        Q = h(d2)
        u = (np.exp((r - q) * dt) * h(d1)) / Q
        d = (np.exp((r - q) * dt) - Q * u) / (1 - Q)

        qu = (np.exp(r * dt) - d) / (u - d)

        qd = 1 - qu

        j = n

        for i in range(j + 1):

            stock_value[j, i] = s0 * u**i * d**(j - i)
            option_value[j, i] = np.maximum(k - stock_value[j, i], 0)

        for j in range(n - 1, -1, -1):

            for i in range(j, -1, -1):

                stock_value[j, i] = s0 * u**i * d**(j - i)

                pv_hold = np.exp(-r * dt) * (qu * option_value[j + 1, i + 1] +
                                             qd * option_value[j + 1, i])
                pv_exercise = np.maximum(k - stock_value[j, i], 0)

                option_value[j, i] = np.maximum(pv_hold, pv_exercise)

        output = {'num_steps': n, 'LRA': option_value[0, 0]}

        lr_result.append(output)

    return (lr_result)


# %%
t_start = tm.datetime.now()
result = lra(s0, k, t, r, sigma, start, n)
t_end = tm.datetime.now()
print(str(t_end - t_start))

# %%
lra_result = pd.DataFrame.from_dict(result)
lra_result['error_LRA'] = ame_value - lra_result['LRA']
print(lra_result.head())
print(lra_result.tail())
# %%
lra_result.to_csv('../data/hw3_2_lra.csv')

# %%
lra_result = pd.read_csv('../data/hw3_2_lra.csv')
plt.figure(figsize=(80 / 3, 15))
plt.plot(lra_result['num_steps'], lra_result['error_LRA'], 'o', markersize=5)
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Error of Leisen and Reimer Model for American Put')
plt.grid()
plt.show()

# %%
extrp_n = np.arange(51, 501, 50)
extrp_value = [
    (i**2 * lra_result['LRA'][i - 50] -
     (2 * i - 1)**2 * lra_result['LRA'][2 * i - 51]) / (i**2 - (2 * i - 1)**2)
    for i in extrp_n
]
extrp_error = -np.array(extrp_value) + ame_value
extrp_error

# %%
plt.figure(figsize=(80 / 3, 15))
plt.plot(extrp_n,
         extrp_error,
         '-',
         linewidth=10,
         label='Extrapulated Error Terms at N&M')
plt.plot(extrp_n,
         lra_result['error_LRA'][2 * extrp_n - 51],
         linewidth=10,
         label='Unextrapulated Error Terms at M=2N-1')
plt.legend(prop={'size': 30})
plt.xlabel('Extrapolated N')
plt.ylabel('Error')
plt.title(
    'Error of Extrapolated Leisen and Reimer Model Model for American Put')
plt.grid()
plt.show()

# %%[markdown]
# ### (3) Broadie and Detemple Model


# %%
def bda(s0, k, t, r, sigma, start, n):

    bd_result = []

    option_value = np.zeros([n + 1, n + 1])

    stock_value = np.zeros([n + 1, n + 1])

    for n in range(start, n + 1, 1):

        dt = t / n

        u = np.exp(sigma * np.sqrt(dt))

        d = 1 / u

        qu = (np.exp(r * dt) - d) / (u - d)

        qd = 1 - qu

        j = n

        for i in range(j + 1):

            stock_value[j, i] = s0 * u**i * d**(j - i)
            option_value[j, i] = np.maximum(k - stock_value[j, i], 0)

        j = n - 1

        for i in range(j + 1):

            stock_value[j, i] = s0 * u**i * d**(j - i)
            option_value[j, i] = black_scholes('put', stock_value[j, i], k, dt,
                                               r, q, sigma)

        for j in range(n - 2, -1, -1):

            for i in range(j, -1, -1):

                stock_value[j, i] = s0 * u**i * d**(j - i)

                pv_hold = np.exp(-r * dt) * (qu * option_value[j + 1, i + 1] +
                                             qd * option_value[j + 1, i])
                pv_exercise = np.maximum(k - stock_value[j, i], 0)

                option_value[j, i] = np.maximum(pv_hold, pv_exercise)

        output = {'num_steps': n, 'BDA': option_value[0, 0]}

        bd_result.append(output)

    return (bd_result)


# %%
t_start = tm.datetime.now()
result = bda(s0, k, t, r, sigma, start, n)
t_end = tm.datetime.now()
print(str(t_end - t_start))

# %%
bda_result = pd.DataFrame.from_dict(result)
bda_result['error_BDA'] = ame_value - bda_result['BDA']
print(bda_result.head())
print(bda_result.tail())
# %%
bda_result.to_csv('../data/hw3_2_bda.csv')

# %%
bda_result = pd.read_csv('../data/hw3_2_bda.csv')
plt.figure(figsize=(80 / 3, 15))
plt.plot(bda_result['num_steps'], bda_result['error_BDA'], 'o', markersize=5)
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Error of Broadie and Detemple Model for American Put')
plt.grid()
plt.show()

# %%
extrp_n = np.arange(50, 501, 50)
extrp_value = [
    (i * bda_result['BDA'][i - 50] - 2 * i * bda_result['BDA'][2 * i - 50]) /
    (-i) for i in extrp_n
]
extrp_error = -np.array(extrp_value) + ame_value

# %%
plt.figure(figsize=(80 / 3, 15))
plt.plot(extrp_n,
         extrp_error,
         '-',
         linewidth=10,
         label='Extrapulated Error Terms at N&M')
plt.plot(extrp_n,
         bda_result['error_BDA'][2 * extrp_n - 50],
         linewidth=10,
         label='Unextrapulated Error Terms at M=2N')
plt.legend(prop={'size': 30})
plt.xlabel('Extrapolated N')
plt.ylabel('Error')
plt.title('Error of Extrapolated Broadie and Detemple Model for American Put')
plt.grid()
plt.show()

# %%[markdown]
# # Question 3: Continuous Barrier

# %%
start = 50
n = 1000
s0 = 100
k = 100
b = 95
sigma = 0.3
r = 0.1
t = 0.2
q = 0

# %%[markdown]
# ## Revised B-S Model


# %%
def bs_barrier_call(s0, k, b, t, r, q, sigma):

    cdf_mean = 0.0
    cdf_sd = 1.0

    d1 = (np.log(s0 / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = (np.log(s0 / k) + (r - q - 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))

    h1 = (np.log(b**2 / (k * s0)) +
          (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    h2 = (np.log(b**2 / (k * s0)) +
          (r - q - 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))

    call0 = s0 * np.exp(-q * t) * st.norm.cdf(d1, cdf_mean, cdf_sd)
    call111111 = call0 - k * np.exp(-r * t) * st.norm.cdf(d2, cdf_mean, cdf_sd)
    call2 = call111111 - (b / s0)**(
        1 + 2 * r * sigma**(-2)) * s0 * st.norm.cdf(h1, cdf_mean, cdf_sd)
    call = call2 + (b / s0)**(-1 + 2 * r * sigma**(-2)) * k * np.exp(
        -r * t) * st.norm.cdf(h2, cdf_mean, cdf_sd)

    return call


# %%
bsbc_value = bs_barrier_call(s0, k, b, t, r, q, sigma)
bsbc_value


# %%
def crrb(s0, k, b, t, r, sigma, start, n):

    crr_result = []

    option_value = np.zeros([n + 1, n + 1])

    stock_value = np.zeros([n + 1, n + 1])

    for n in range(start, n + 1, 1):

        dt = t / n

        u = np.exp(sigma * np.sqrt(dt))

        d = 1 / u

        qu = (np.exp(r * dt) - d) / (u - d)

        qd = 1 - qu

        j = n

        for i in range(j + 1):

            stock_value[j, i] = s0 * u**i * d**(j - i)

            if stock_value[j, i] < b:
                option_value[j, i] = 0

            else:
                option_value[j, i] = np.maximum(stock_value[j, i] - k, 0)

        for j in range(n - 1, -1, -1):

            for i in range(j, -1, -1):

                stock_value[j, i] = s0 * u**i * d**(j - i)

                if stock_value[j, i] < b:
                    option_value[j, i] = 0

                else:
                    option_value[j, i] = np.exp(
                        -r * dt) * (qu * option_value[j + 1, i + 1] +
                                    qd * option_value[j + 1, i])

        su = min(list(x for x in stock_value[n] if x > b))
        sd = max(list(x for x in stock_value[n] if x < b))

        lbd = (su - b) / (su - sd)

        output = {'num_steps': n, 'CRRB': option_value[0, 0], 'lambda': lbd}

        crr_result.append(output)

    return (crr_result)


# %%
t_start = tm.datetime.now()
result = crrb(s0, k, b, t, r, sigma, start, n)
t_end = tm.datetime.now()
print(str(t_end - t_start))

# %%
crrb_result = pd.DataFrame.from_dict(result)
crrb_result['error_CRRB'] = bsbc_value - crrb_result['CRRB']
print(crrb_result.head())
print(crrb_result.tail())
# %%
crrb_result.to_csv('../data/hw3_3_crrb.csv')

# %%
crrb_result = pd.read_csv('../data/hw3_3_crrb.csv')
plt.figure(figsize=(80 / 3, 15))
plt.plot(crrb_result['num_steps'],
         crrb_result['error_CRRB'],
         'o',
         markersize=5)
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Error of CRR Model for Continuous Barrier Call Option')
plt.grid()
plt.show()

# %%
crrb_result = pd.read_csv('../data/hw3_3_crrb.csv')
plt.figure(figsize=(80 / 3, 15))
plt.plot(crrb_result['num_steps'], crrb_result['lambda'], 'o', markersize=5)
plt.xlabel('N')
plt.ylabel('Lambda')
plt.title('Lambda of CRR Model for Continuous Barrier Call Option')
plt.grid()
plt.show()

# %%[markdown]
# # Question 4: Discrete Barrier

# %%
real_value = 5.6711051343


# %%
def crrdb(s0, k, b, t, r, sigma, start, n):

    crr_result = []

    option_value = np.zeros([n + 1, n + 1])

    stock_value = np.zeros([n + 1, n + 1])

    for n in range(start, n + 1, 10):

        dt = t / n

        u = np.exp(sigma * np.sqrt(dt))

        d = 1 / u

        qu = (np.exp(r * dt) - d) / (u - d)

        qd = 1 - qu

        j = n

        for i in range(j + 1):

            stock_value[j, i] = s0 * u**i * d**(j - i)

            option_value[j, i] = np.maximum(stock_value[j, i] - k, 0)

        for j in range(n - 1, -1, -1):

            if j in [n / 5, 2 * n / 5, 3 * n / 5, 4 * n / 5]:

                for i in range(j, -1, -1):

                    stock_value[j, i] = s0 * u**i * d**(j - i)

                    if stock_value[j, i] < b:
                        option_value[j, i] = 0

                    else:
                        option_value[j, i] = np.exp(
                            -r * dt) * (qu * option_value[j + 1, i + 1] +
                                        qd * option_value[j + 1, i])

            else:

                for i in range(j, -1, -1):

                    stock_value[j, i] = s0 * u**i * d**(j - i)
                    option_value[j, i] = np.exp(
                        -r * dt) * (qu * option_value[j + 1, i + 1] +
                                    qd * option_value[j + 1, i])

        su = min(list(x for x in stock_value[n, :] if x > b))
        sd = max(list(x for x in stock_value[n, :] if x < b))

        lbd = (su - b) / (su - sd)

        output = {'num_steps': n, 'CRRDB': option_value[0, 0], 'lambda': lbd}

        crr_result.append(output)

    return (crr_result)


# %%
t_start = tm.datetime.now()
result = crrdb(s0, k, b, t, r, sigma, start, n)
t_end = tm.datetime.now()
print(str(t_end - t_start))

# %%
crrdb_result = pd.DataFrame.from_dict(result)
crrdb_result['error_CRRDB'] = real_value - crrdb_result['CRRDB']
print(crrdb_result.head())
print(crrdb_result.tail())
# %%
crrdb_result.to_csv('../data/hw3_4_crrdb.csv')

# %%
crrbd_result = pd.read_csv('../data/hw3_4_crrdb.csv')
plt.figure(figsize=(80 / 3, 15))
plt.plot(crrdb_result['num_steps'],
         crrdb_result['error_CRRDB'],
         'o',
         markersize=5)
plt.xlabel('N')
plt.ylabel('Error')
plt.title('Error of CRR Model for Discrete Barrier Call Option')
plt.grid()
plt.show()

# %%
crrbd_result = pd.read_csv('../data/hw3_4_crrdb.csv')
plt.figure(figsize=(80 / 3, 15))
plt.plot(crrdb_result['num_steps'], crrdb_result['lambda'], 'o', markersize=5)
plt.xlabel('N')
plt.ylabel('Lambda')
plt.title('Lambda of CRR Model for Discrete Barrier Call Option')
plt.grid()
plt.show()

# %%
