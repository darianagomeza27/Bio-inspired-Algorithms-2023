#!/usr/bin/env python
# coding: utf-8

# # Bio-inspired Algorithms
# ## Dariana Gomez

# In[1]:


def mapping(x0_, x1_, n_):
    print(f"Mapping for range [{x0_}, {x1_}]:\n")
    print(f"{'i':<5}{'   y':<10}")
    print("-" * 15)
    
    for i in range(2**n_):
        y = x0_ + i * (x1_ - x0_) / (2**n_ - 1)
        print(f"{i:<5}{y:<10.5f}")


# ### [0,5]

# In[2]:


x0 = 0 # initial value
x1 = 5 # end value
n = 8 # bits

mapping(x0,x1,n)


# ### [0,1]

# In[3]:


x0 = 0 # initial value
x1 = 1 # end value
n = 8 # bits

mapping(x0,x1,n)


# ### [-1,1]

# In[4]:


x0 = -1 # initial value
x1 = 1 # end value
n = 8 # bits

mapping(x0,x1,n)


# ### [-10,5]

# In[5]:


x0 = -10 # initial value
x1 = 5 # end value
n = 8 # bits

mapping(x0,x1,n)

