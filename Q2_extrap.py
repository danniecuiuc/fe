#!/usr/bin/env python
# coding: utf-8

# Spring 2020
# 
# This notebook provides the extrapolation part of Q2. This gives you (and me) a chance to try to understand data frames.
# 
# Packages and Configurations
# The following common packages will be use on this notebook.
# 
# numpy - https://numpy.org/
# Pandas - https://pandas.pydata.org/
# matplotlib - https://matplotlib.org/
# Scipy Statistical functions - https://docs.scipy.org/doc/scipy/reference/stats.html
# import numpy as np
# import pandas as pd
# import scipy.stats as st
# import matplotlib.pyplot as plt
# 

# In[27]:


import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt


# In[46]:


# Enter the smallest number of steps considered (this will be the first entry in num_steps)
#For CRR and RB this is 50, for LR this is 51
nstart = 50


# In[47]:


# ENTER INPUT FOR LOW RANGE OF EXTRAPOLATION, NMIN
#For CRR and RB this is 50, for LR this is 51
nmin = 50


# In[48]:


# # ENTER INPUT FOR HIGH RANGE OF EXTRAPOLATION, NMAX
#For CRR and RB this is 500, for LR this is 501
nmax = 500


# In[49]:


# ENTER THE STEPS BETWEEN N VALUES
step = 50


# In[50]:


# LR EXTRAPOLATION IS DIFFERENT SO ENTER LR = 1 IF YOU ARE USING LR DATA AND 0 IF NOT
lr = 0


# In[51]:


# Read in the appropriate data from the csv file
df = pd.read_csv("Data/Q2crra.csv")


# In[52]:


# Check that the data file looks correct
df


# In[54]:


# Now let's just focus on the values, 
# the "CRR" will be the column of your data with the option values that you're going to interpolate with

value = df["CRR"]
value


# In[55]:


def extrap(value, nmin, nmax, step, lr):
    

    if lr == 1:
        nmaxnew = int(1+((2*nmax)-3-nstart)/2)
        n_steps = list(range(nmin-nstart,nmaxnew))
        d = value[n_steps].to_dict()
        result = []
        for i in range (nmin,nmax+step,step):
            high = 2*(i-1)-1
            low = i
            n1 = (high-nstart)/2
            n2 = (low-nstart)/2
            extrap = (high*d[n1]-low*d[n2])/(high-low)
            output = {'num_steps': low, 'extrap': extrap}
            result.append(output)
        return result
    else:
        n_steps = list(range(nmin-nstart,2*nmax+1-nstart+step))
        d = value[n_steps].to_dict()
        result = []
        for i in range (nmin,nmax+step,step):
            high = 2*i
            low = i
            n1 = high-nstart
            n2 = low-nstart
            extrap = (high*d[n1]-low*d[n2])/(high-low)
            output = {'num_steps': low, 'extrap': extrap}
            result.append(output)
        return result
                            


# In[56]:


results = extrap(value,nmin,nmax,step,lr)
results


# In[57]:



df = pd.DataFrame.from_dict(results)
df


# In[58]:


df.to_csv("Data/Q2_crrextrap.csv", index=False)


# ### Binomial Model Error Rate

# In[59]:


plt.figure(figsize=(14,10))
plt.plot(df['num_steps'], df['extrap'], 'o', markersize=3)
plt.savefig('Images/Q2_crrextrap.png')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




