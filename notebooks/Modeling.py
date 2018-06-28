
# coding: utf-8

# In[13]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import scipy as sp 
import statsmodels.stats as stats
import statsmodels.api as sm 
from patsy import dmatrix
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', "InlineBackend.figure_formats = {'png', 'retina'}")
import pickle 
#pd.options.display.max_rows = 100
#pd.options.display.max_columns = 100
#np.set_printoptions(threshold=np.inf) # print numpy array fully 

train_16 = pd.read_csv("../Data/train_2016_v2.csv")
properties_16 = pd.read_csv("../Data/properties_2016.csv")

# intergrating dataset : properties_16 : left, train_16 : right, left join 
df_16 = pd.merge(properties_16, train_16, on="parcelid", how="inner")


# In[14]:


# categorical 로 처리 할 컬럼과 Numerical로 처리할 컬럼 구분 
catecols = []
numecols = []
obj_type = df_16.hashottuborspa.dtype

for col in df_16.columns:
    if df_16[col].dtype == obj_type:
        catecols.append(col)
    elif len(np.unique(df_16[col].fillna(0))) <= 30:
        catecols.append(col)
    else : 
        numecols.append(col)
len(catecols), len(numecols)


# In[15]:


print(len(catecols), len(numecols))
catecols.remove('transactiondate')
numecols.remove('parcelid')
numecols.remove('logerror')
print(len(catecols), len(numecols))


# In[16]:


# data imputaion 
# cate -> 0 
# numerical -> np.mean ## 아주 간단한 버전 ! 
df_16[catecols] = df_16[catecols].fillna(0)
df_16[numecols] = df_16[numecols].fillna(np.mean)


# In[17]:


# categorical data processing 
def cate(col_list):
    return ["C({})".format(x) for x in col_list]


# In[18]:


# globals()


# In[ ]:


input_cols = cate(catecols) + numecols
# formular = "logerror ~ " + (" + ").join(input_cols)
# formular


# In[ ]:


# 기본 모델링 with OLS

model_init = sm.regression.linear_model.OLS.from_formula("logerror ~ " + (" + ").join(input_cols), data=df_16)
model_init = model_init.fit()
print(model_init.summary())

