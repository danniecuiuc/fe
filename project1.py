import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt

start_step =100
N = 1000
S0 = 159.98
sigma1 = 0.47089
sigma2 = 0.39659
sigma3 = 0.365
sigma4 = 0.34767
sigma5 = 0.34724
r1 = 0.01836
r2 = 0.01720
r3 = 0.01650
r4 = 0.01592
r5 = 0.015918
r12 = 0.011055
r23 = 0.008214
r34 = 0.006374
r45 = 0.005919
div = 0.00406
T = 0.2
K=119.185


def up(delta,sigma):
    u = np.exp(sigma * (delta)**0.5)
    return u

def period_stock_price(S0 ,j,i,delta, sigma):
    u = up(delta,sigma)
    d = 1/u
    stock_value = S0 * (u ** i) * (d ** (j - i))
    return stock_value

def period_qu_qd(r, delta,u):
    qu = (np.exp(r * delta) - 1/u) / (u - 1/u)
    qd = 1 - qu
    return (qu,qd)



def CRR_autocall_model(S0, K, T, r1,r2,r3,r4,r5,r12,r23,r34,r45, sigma1,sigma2, sigma3, sigma4,sigma5, start_step, N):


    # LIST TO SAVE RESULTS
    result = []
    note_value = np.zeros([ N + 1, N + 1])

    # CREATE ARRAY FOR STOCK PRICES OF SIZE N+1,N+1
    stock_value = np.zeros([ N + 1,N + 1])

    # FOR LOOP STATEMENT: For a Binomial Tree from start_step to N
    for n1 in range(start_step, N + 1):

        n = n1
        j1 = n1 * 0.2453
        j2 = n1 * 0.4906
        j3 = n1 * 0.7439
        j4 = n1 * 0.9919
        delta = T / n1
        for i1 in range(0, round(j1) + 1):
            stock_value[i1, round(j1)] = period_stock_price(S0=S0, j=round(j1), i=i1, delta=delta, sigma=sigma1)
        for i2 in range(0, round(j2) + 1):
            if i2 < round(j2) - round(j1):
                stock_value[i2, round(j2)] = period_stock_price(S0=stock_value[0, round(j1)],j=round(j2) - round(j1) - i2, i=i2, delta=delta,sigma=sigma2)
            else:
                stock_value[i2, round(j2)] = stock_value[i2 + round(j1) - round(j2), round(j1)] * up(delta, sigma2) ** (round(j2) - round(j1))
        for i3 in range(0, round(j3) + 1):
            if i3 < round(j3) - round(j2):
                stock_value[i3, round(j3)] = period_stock_price(S0=stock_value[0, round(j2)],j=round(j3) - round(j2) - i3, i=i3, delta=delta,sigma=sigma3)
            else:
                stock_value[i3, round(j3)] = stock_value[i3 + round(j2) - round(j3), round(j2)] * up(delta, sigma3) ** (round(j3) - round(j2))
        for i4 in range(0, round(j4) + 1):
            if i4 < round(j4) - round(j3):
                stock_value[i4, round(j4)] = period_stock_price(S0=stock_value[0, round(j3)],j=round(j4) - round(j3) - i4, i=i4, delta=delta,sigma=sigma4)
            else:
                stock_value[i4, round(j4)] = stock_value[i4 + round(j3) - round(j4), round(j3)] * up(delta, sigma4) ** (round(j4) - round(j3))
        for i5 in range(0, n1 + 1):
            if i5 < n1 - round(j4):
                stock_value[i5, n1] = period_stock_price(S0=stock_value[0, round(j4)], j=n1 - round(j4) - i5, i=i5,delta=delta, sigma=sigma5)
            else:
                stock_value[i5, n1] = stock_value[i5 + round(j4) - n1, round(j4)] * up(delta, sigma5) ** (n1 - round(j4))
            note_value[i5, n1] = 10.2375 if stock_value[i5, n1] >= 119.985 else 10 * stock_value[i5, n1] / 159.98



        for q5 in range(n1-1,round(j4)-1,-1):
            for i in range(q5,-1,-1):
                qu = period_qu_qd(r45, delta=delta, u=up(delta,sigma5))[0]
                qd= 1-qu
                pv = np.exp(-r5 * delta) * (qu * note_value[i + 1, q5 + 1] + qd * note_value[i, q5+1])
                if q5 == round(j4):
                    note_value[i,q5] = 10.2375 if stock_value[i,q5] >= S0 else pv
                else:
                    note_value[i, q5] = pv

        for q4 in range(round(j4)-1, round(j3)-1,-1):
            for i in range(q4,-1,-1):
                qu = period_qu_qd(r34, delta=delta, u=up(delta,sigma4))[0]
                qd = 1 - qu
                pv = np.exp(-r4 * delta) * (qu * note_value[i + 1, q4 + 1] + qd * note_value[i, q4+1])
                if q4 == round(j3):
                    note_value[i, q4] = 10.2375 if stock_value[i, q4] >= S0 else pv
                else:
                    note_value[i, q4] = pv

        for q3 in range(round(j3)-1, round(j2)-1,-1):
            for i in range(q3,-1,-1):
                qu = period_qu_qd(r23, delta=delta, u=up(delta,sigma3))[0]
                qd = 1 - qu
                pv = np.exp(-r3 * delta) * (qu * note_value[i + 1, q3 + 1] + qd * note_value[i, q3+1])
                if q3 == round(j2):
                    note_value[i, q3] = 10.2375 if stock_value[i, q3] >= S0 else pv
                else:
                    note_value[i, q3] = pv

        for q2 in range(round(j2)-1, round(j1)-1,-1):
            for i in range(q2,-1,-1):
                qu = period_qu_qd(r12, delta=delta, u=up(delta,sigma2))[0]
                qd = 1 - qu
                pv = np.exp(-r2 * delta) * (qu * note_value[i + 1, q2 + 1] + qd * note_value[i, q2+1])
                note_value[i, q2] = pv
                if q2 == round(j1):
                    note_value[i, q2] = 10.2375 if stock_value[i, q2] >= S0 else pv
                else:
                    note_value[i, q2] = pv

        for q1 in range(round(j1),-1,-1):
            for i in range(q1,-1,-1):
                qu = period_qu_qd(r1, delta=delta, u=up(delta,sigma1))[0]
                qd = 1 - qu
                pv = np.exp(-r1 * delta) * (qu * note_value[i + 1, q1 + 1] + qd * note_value[i, q1+1])
                note_value[i, q1] = pv


        # RELAY OUTPUTS TO DICTIONARY
        output = {'num_steps': n, 'CRR': note_value[0, 0]}
        result.append(output)

    return result

crr_autocall = CRR_autocall_model(S0, K, T, r1,r2,r3,r4,r5,r12,r23,r34,r45, sigma1,sigma2, sigma3, sigma4,sigma5, start_step, N)
df = pd.DataFrame.from_dict(crr_autocall)

plt.plot(df['num_steps'], df['CRR'],"o", markersize=3, color = 'tab:red')
plt.ylabel('note value (CRR)')
plt.xlabel('size of binomial tree')
plt.title("note value based on CRR model")
plt.show()

