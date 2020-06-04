import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

start_step =100
N = 200
S0 = 159.98
downside_threshold = 19.185
# sigma based on the period between the pricing date and maturity and 75% moneyness
sigma = 0.31175
r= 0.015918
div = 0.4
T = 0.2
K=119.185




def CRR_model(S0, K,  T, r, sigma, start_step, N):


    # LIST TO SAVE RESULTS
    crr_result = []

    # CREATE TWO DIMENSIONAL ARRAY OF SIZE [N+1,N+1] TO STORE ALL STEPS
    # option_value[N+1, N+1]
    note_value = np.zeros([10 * N + 1, 10 * N + 1])

    # CREATE ARRAY FOR STOCK PRICES OF SIZE N+1,N+1
    # stock_value[N+1, N+1]
    stock_value = np.zeros([10 * N + 1, 10 * N + 1])

    # FOR LOOP STATEMENT: For a Binomial Tree from start_step to N
    for n1 in range(start_step, N + 1):

        n = n1
        j1 = n1 * 0.2453
        j2 = n1 * 0.4906
        j3 = n1 * 0.7439
        j4 = n1 * 0.9919

        delta = T / n
        u = np.exp(sigma * (delta) ** 0.5)
        d = 1 / u
        qu = (np.exp(r * delta) - d) / (u - d)
        qd = 1 - qu

        j = n

        for i in range(0, j + 1):
            stock_value[j, i] = S0 * (u ** i) * (d ** (j - i))
            note_value[j, i] = 10.2375 if stock_value[j,i]>=119.985 else 10*stock_value[j, i]/159.98


        for j in range(n - 1, -1, -1):

            for i in range(j, -1, -1):

                # Now, calculation the PV of the option values at that specific position and step number
                # Hint: V = e^(-r x Delta) (qu x Vup + qd x Vdown)
                pv = np.exp(-r * delta) * (qu * note_value[j + 1, i + 1] + qd * note_value[j + 1, i])
                stock_value[j, i] = S0 * (u ** i) * (d ** (j - i))
                if (j == round(j1) or j == round(j2) or j == round(j3) or j == round(j4)):
                    if stock_value[j, i] >= S0:
                        note_value[j, i] = 10.2375
                    else:
                        note_value[j, i] = pv
                else:
                    note_value[j, i] = pv
        output = {'num_steps': n, 'CRR': note_value[0, 0]}
        crr_result.append(output)


    return crr_result


crr_autocall = CRR_model(S0,K,T,r,sigma,start_step, N)
df = pd.DataFrame.from_dict(crr_autocall)

plt.plot(df['num_steps'], df['CRR'], markersize=3, color = 'tab:red')
plt.ylabel('Error')
plt.show()