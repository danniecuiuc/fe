#!/usr/bin/env python
# coding: utf-8

# # FIN 514 - PS3 Q1 CRR
# **Spring 2020**
# 
# This notebook provides the graphs for PS3 Q1 for the CRR model
# 
# ## Packages and Configurations
# 
# The following common packages will be use on this notebook.
# 
# * numpy - [https://numpy.org/](https://numpy.org/)
# * Pandas - [https://pandas.pydata.org/](https://pandas.pydata.org/)
# * matplotlib - [https://matplotlib.org/](https://matplotlib.org/)
# * Scipy Statistical functions - [https://docs.scipy.org/doc/scipy/reference/stats.html](https://docs.scipy.org/doc/scipy/reference/stats.html)
# 

# In[1]:


import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt


# 

# 
# 

# 

# In[2]:


# ENTER INPUT FOR: start_step

start_step = 50


# In[3]:


# ENTER INPUT FOR: N = num_steps

N = 1000


# In[4]:


# ENTER INPUT FOR: S0 = Original Stock Price

S0 = 100.0


# In[5]:


# ENTER INPUT FOR: K = Excercise Price of Call Option

K = 95.0


# In[6]:


# ENTER INPUT FOR: sigma = Annualized (Future) Volatility of Stock Price Returns

sigma = 0.3


# In[7]:


# ENTER INPUT FOR: r = Annualized Continously Compounded Risk-free Rate

r = 0.1


# In[8]:


# ENTER INPUT FOR: T = Time Length of Option in which to Exercise (In Years)

T = 0.2


# In[9]:


# ENTER INPUT FOR: q = Rate of continuous dividend paying asset 

q = 0


# In[10]:


# ENTER INPUT FOR: whether option is call (1) or put (0)
cp = 0


# ## Black-Sholes Model 

# In[11]:


def black_scholes(S0, K, T, r, q, sigma, cp):
    """
    Function to calculates the value of a European Call Option using Black Scholes 
    
    S0: Original Stock Price
    K: Excercise Price of Call Option
    T: Time Length of Option in which to Exercise (In Years)
    r: Annualized Continously Compounded Risk-free Rate
    q: Rate of continuous dividend paying asset 
    sigma: Annualized (Future) Volatility of Stock Price Returns
    
    """
    
    cdf_mean = 0.0
    cdf_sd = 1.0
    
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = (np.log(S0 / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    
    if cp == 1:
        value = S0 * np.exp(-q * T) * st.norm.cdf(d1, cdf_mean, cdf_sd) 
        value = value - K * np.exp(-r * T) * st.norm.cdf(d2, cdf_mean, cdf_sd) 
    else:
        value = -S0 * np.exp(-q * T) * st.norm.cdf(-d1, cdf_mean, cdf_sd) 
        value = value + K * np.exp(-r * T) * st.norm.cdf(-d2, cdf_mean, cdf_sd) 
    
    return value


# ## Binomial Model Function

# In[12]:



def CRRE_model(S0, K, T, r, sigma, start_step, N):
    """
    Function to calculates the value of a European Put Option using the CRR Binomial Model 
    
    S0: Original Stock Price
    K: Excercise Price of Call Option
    T: Time Length of Option in which to Exercise (In Years)
    r: Annualized Continously Compounded Risk-free Rate
    sigma: Annualized (Future) Volatility of Stock Price Returns
    start_step: Starting time step
    N: Number of time steps
    
    """    
    
    # LIST TO SAVE RESULTS
    crre_result = []
        
    # CREATE TWO DIMENSIONAL ARRAY OF SIZE [N+1,N+1] TO STORE ALL STEPS
    # option_value[N+1, N+1]
    option_value = np.zeros([N+1, N+1])

    # CREATE ARRAY FOR STOCK PRICES OF SIZE N+1,N+1
    # stock_value[N+1, N+1]
    stock_value = np.zeros([N+1, N+1])    
    
    # FOR LOOP STATEMENT: For a Binomial Tree from start_step to N
    for n in range(start_step, N+1):
        delta = T / n
        u = np.exp(sigma * (delta)**0.5)
        d = 1 / u
        qu = (np.exp(r * delta) - d) / (u - d)
        qd = 1 - qu
        
    # CALCULATE OPTION VALUES AT CERTAIN STEPS AND POSITIONS WITHIN THE BINOMIAL TREE:
    # Start at the last step number because we are going to be moving backwards from step number n to step number 0
    # Hint: j = n and range stop = j 
        j = n 
        
        for i in range(0, j):    
    # Then, calculate the value of the option at that exact position within the binomial tree
    # REMEMBER: the value of the option is MAX(stock_value - Exercise Price, 0)
    # Hint: V = np.maximum(S - K, 0)
            stock_value[j, i] = S0 * (u**i) * (d**(j - i))
            option_value[j, i] = np.maximum(K - stock_value[j, i], 0)

    # Now, lets calculate the option value at each position (i) within the binomial tree at each previous step number (j) until time zero
    # First, start with a FOR iteration on the step number
    # Step backwards (Step -1) for the step number (j) because you are working backwards from the 2nd to last step (j - 1) to step number 0 
    # Hint: start = (Step -1), stop = -1 (end of range is exclusive, stops at 0 when stop=-1) , step = -1 (moving backwards)
        for j in range(n-1, -1, -1):

    # Then, create a FOR iteration on the position number (i), from the top position all the way down to the bottom position of 0 (all down jumps)
    # Hint: the top positions always equals j (the maximum number of possible up jumps at any time)
    # Hint: use Step -1, since you are moving from the top to the bottom. stop = -1 (end of range is exclusive, stops at 0 when stop=-1)        
            for i in range(j, -1, -1):
            
    # Now, calculation the PV of the option values at that specific position and step number
    # Hint: V = e^(-r x Delta) (qu x Vup + qd x Vdown)            
                pv = np.exp(-r * delta) * (qu * option_value[j + 1, i + 1] + qd * option_value[j + 1, i])
                option_value[j, i] = pv
    # RELAY OUTPUTS TO DICTIONARY
        output = {'num_steps': n, 'CRR': option_value[0,0]}
        crre_result.append(output)

    return crre_result


# In[13]:


crr = CRRE_model(S0, K, T, r, sigma, start_step, N)


# In[14]:


bsc_value = black_scholes(S0, K, T, r, q, sigma,cp)
bsc_value


# In[15]:


# CREATE A DATAFRAME FROM THE BINOMIAL MODEL OUTPUT
df = pd.DataFrame.from_dict(crr)


# In[16]:


# CALCULATE THE ERROR FROM BINOMIAL MODEL COMPARED WITH BLACK-SHCOLES
df['error_CRR'] = df["CRR"] - bsc_value 


# In[17]:


# INSPECT THE FIRST ROWS OF THE DATAFRAME
df.head()


# In[18]:


# INSPECT THE LAST ROWS OF THE DATAFRAME
df.tail()


# In[19]:


# EXPORT THE DATA TO A CSV FILE
df.to_csv("Data/Q1crre.csv", index=False)


# ### Binomial Model Error Rate

# In[20]:


plt.figure(figsize=(14,10))
plt.plot(df['num_steps'], df['error_CRR'], 'o', markersize=3)
plt.savefig('Images/Q1_crre.png')
plt.show()


# Now let's try some extrapolation

# In[ ]:




