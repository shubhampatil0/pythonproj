#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import os


# In[2]:


os.chdir (r'H:\MTECH\MTP')


# In[3]:


import pandas as pd
import numpy as np


# In[4]:


data = pd.read_excel('vel vs time.xlsx')


# In[5]:


data.head(10)


# In[18]:


xdata=np.array(list(data['x']))
ydata= np.array(list(data['y']))


# In[19]:


xdata


# In[20]:


ydata


# In[21]:


curve= np.polyfit(xdata,ydata,3)
poly =np.poly1d(curve)
print(poly)


# In[22]:


print(poly(1.18345165e-01))


# In[23]:


def func(x,a,b,c,d):
    return a*np.sin(b*x)+c*np.cos(d*x)


# In[17]:


get_ipython().system('pip install scikit-learn')


# In[25]:


from scipy.optimize import curve_fit


# In[26]:


popt,pcov = curve_fit(func,xdata,ydata)
print(popt)      #a,b,c,d


# In[27]:


print(pcov) #covariance


# In[ ]:




