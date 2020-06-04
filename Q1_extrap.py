#!/usr/bin/env python
# coding: utf-8

# Spring 2020
# 
# This notebook provides the extrapolation part of Q1. This gives you (and me) a chance to try to understand data frames.
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

# In[147]:


import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt


# In[148]:


# Enter the smallest number of steps considered (this will be the first entry in num_steps)
#For CRR and RB this is 50, for LR this is 51
nstart = 51


# In[149]:


# ENTER INPUT FOR LOW RANGE OF EXTRAPOLATION, NMIN
#For CRR and RB this is 50, for LR this is 51
nmin = 51


# In[150]:


# # ENTER INPUT FOR HIGH RANGE OF EXTRAPOLATION, NMAX
#For CRR and RB this is 500, for LR this is 501
nmax = 501


# In[151]:


# ENTER THE STEPS BETWEEN N VALUES
step = 50


# In[152]:


# LR EXTRAPOLATION IS DIFFERENT SO ENTER LR = 1 IF YOU ARE USING LR DATA AND 0 IF NOT
lr = 1


# In[153]:


# Read in the appropriate data from the csv file
df = pd.read_csv("Data/Q1lre.csv")


# In[154]:


# Check that the data file looks correct
df


# In[155]:


# Now let's just focus on the values, 
# the "CRR" will be the column of your data with the option values that you're going to interpolate with

value = df["LR"]
value


# In[161]:



((2*nmax)-3-nstart)/2
    


# In[162]:


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
                            


# In[163]:


results = extrap(value,nmin,nmax,step,lr)
results


# In[164]:



df = pd.DataFrame.from_dict(results)
df


# In[165]:


df.to_csv("Data/Q1_lrextrap.csv", index=False)


# ### Binomial Model Error Rate

# In[166]:


plt.figure(figsize=(14,10))
plt.plot(df['num_steps'], df['extrap'], 'o', markersize=3)
plt.savefig('Images/Q1_lrextrap.png')
plt.show()


# In[ ]:




