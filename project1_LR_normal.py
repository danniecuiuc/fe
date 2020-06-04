import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

start_step =100
N = 200
S0 = 159.98
sigma = 0.34724
r= 0.015918
## discrete div
div = 0.00406/90
T = 0.2
K=119.185



def LR_model(S0, K, T, r, sigma, start_step, N,div):

    lr_result = []

    note_value = np.zeros([N + 1, N + 1])

    stock_value = np.zeros([N + 1, N + 1])

    for n1 in range(start_step, N + 1,1):


        n = n1
        j1 = n1 * 0.2453
        j2 = n1 * 0.4906
        j3 = n1 * 0.7439
        j4 = n1 * 0.9919

        delta = T / n
        d1 = (np.log(S0 / K) + (r - div + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = (np.log(S0 / K) + (r - div - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        qu = 0.5 + np.sqrt(0.25 - 0.25 * np.exp(-(d2 / (n + 1 / 3)) ** 2 * (n + 1 / 6)))
        if d2 < 0:
            qu = 0.5 - np.sqrt(0.25 - 0.25 * np.exp(-(d2 / (n + 1 / 3)) ** 2 * (n + 1 / 6)))
        qd = 1 - qu
        qstar = 0.5 + np.sqrt(0.25 - 0.25 * np.exp(-(d1 / (n + 1 / 3)) ** 2 * (n + 1 / 6)))
        if d1 < 0:
            qstar = 0.5 - np.sqrt(0.25 - 0.25 * np.exp(-(d1 / (n + 1 / 3)) ** 2 * (n + 1 / 6)))
        u = np.exp((r - div) * delta) * qstar / qu
        d = (np.exp((r - div) * delta) - qu * u) / (1 - qu)

        j = n

        for i in range(0, j):
            stock_value[j, i] = S0 * (u ** i) * (d ** (j - i))
            note_value[j, i] = 10.2375 if stock_value[j,i]>=119.985 else 10*stock_value[j, i]/159.98


        for j in range(n - 1, -1, -1):

            for i in range(j, -1, -1):

                pv = np.exp(-r * delta) * (qu * note_value[j + 1, i + 1] + qd * note_value[j + 1, i])
                stock_value[j, i] = S0 * (u ** i) * (d ** (j - i))
                if (j == round(j1) or j==round(j2) or j== round(j3) or j==round(j4)):
                   if stock_value[j,i] >= S0:
                       note_value[j,i] = 10.2375
                   else:
                       note_value[j,i] = pv
                else:
                    note_value[j,i] = pv

        output = {'num_steps': n, 'LR': note_value[0, 0]}
        lr_result.append(output)

    return lr_result


lr_normal = LR_model(S0, K, T, r, sigma, start_step, N, div)
df = pd.DataFrame.from_dict(lr_normal)
df['error_LR'] = df["LR"]



plt.plot(df['num_steps'], df['error_LR'] ,markersize=3, color = 'tab:red')
plt.ylabel('Error')
plt.show()

