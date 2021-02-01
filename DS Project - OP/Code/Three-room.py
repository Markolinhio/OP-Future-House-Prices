#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.impute import SimpleImputer
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pandas.testing as tm
from matplotlib import pyplot
import statsmodels.api as sm
import plotly.graph_objects as go


# In[3]:


get_ipython().system('pip install fbprophet')


# In[4]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from fbprophet import Prophet
from pandas import to_datetime
import datetime


# In[5]:


get_ipython().system('pip install plotly==4.11.0 ')


# In[6]:


df = pd.read_csv("ThreeRoom.csv", sep = ';', encoding='cp437')
data = df.replace('...', np.nan).drop([ "Building type", "Number of rooms"], axis =1)
data.index = data["Region"]
data = data.drop(["Region"], axis = 1)
data = data.drop(["Uusimaa", "Helsinki"])
data


# In[7]:


data2 = data.reset_index(drop=True)

for c in data2.columns:
    data2[str(c)] = pd.to_numeric(data2[str(c)], errors='coerce')
    
data2.index = data.index
data2


# In[8]:


error = data2.diff(axis=1)
values1 = []

for r in error.index:
    for q in error.columns:
        if abs(float(error.at[str(r), str(q)])) > 500:
            values1.append(tuple([str(r), str(q), error.at[str(r), str(q)]])) 


# In[9]:


missing_values_count = data.isnull().sum(axis=1)
print(missing_values_count)
indexes3 = data.index.tolist()
deleted_rows = []
i = 0

      
for row in missing_values_count:
    if row > 0.7*59:
        deleted_rows.append(str(indexes3[i]))
    i = i + 1

print(deleted_rows)
print(str(len(deleted_rows)) + " regions have less than 30% of data filled one room flats")


# In[42]:


missing_values_count = data.isnull().sum(axis=1)

total_cells= np.product(data.shape)
total_missing = missing_values_count.sum()

(total_missing/total_cells)*100


# In[43]:


for i in deleted_rows:
    data = data.drop(str(i), axis = 0)

data


# In[44]:


missing_values_count = data.isnull().sum(axis=1)

total_cells= np.product(data.shape)
total_missing = missing_values_count.sum()

(total_missing/total_cells)*100


# In[45]:


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(random_state=0)
final_data = imp.fit_transform(data.to_numpy())
final_data = pd.DataFrame(data = final_data, columns = data.columns)
final_data.index = data.index
final_data


# In[46]:


data1 = final_data.reset_index(drop=True)

for c in data1.columns:
    data1[str(c)] = pd.to_numeric(data1[str(c)], errors='coerce')
    
data1.index = final_data.index
data1


# In[49]:


error = data1.diff(axis=1)
values = []
missing = []

for r in error.index:
    for q in error.columns:
        if abs(float(error.at[str(r), str(q)])) > 500:
            values.append(tuple([str(r), str(q), error.at[str(r), str(q)]]))
            index = index + 1
for i in values:
    if i not in values1:
        missing.append(i)
            
missing


# In[10]:


df = pd.read_csv("ThreeRoomLP", sep = ';', encoding='cp437')
df.index = df["Region"]
df = df.drop(["Region"], axis = 1)
df


# In[11]:


writer = pd.ExcelWriter('ResultsThreeRoomLinear.xlsx')
df.to_excel(writer)
writer.save()


# In[3]:


lst = []
    
dataFrame = pd.DataFrame(lst, index = df.index, columns = df.columns)
  
for region in df.index:

        base = df.loc[region:region,"2018Q1": "2018Q4"]
        baseAverage = base.mean(axis = 1)
        
        for quarter in df.columns:
            CPI = df.loc[region, quarter]/baseAverage[region]
            dataFrame.at[region, quarter] = CPI
            
dataFrame


# In[4]:


writer = pd.ExcelWriter('ThreeRoomLP.xlsx')
dataFrame.to_excel(writer)
writer.save()


# In[12]:


df = pd.read_csv("ThreeRoomProphet", sep = ';', encoding='cp437')
df.index = df["Region"]
df = df.drop(["Region"], axis = 1)
df


# In[13]:


writer = pd.ExcelWriter('ResultsThreeRoomProphet.xlsx')
df.to_excel(writer)
writer.save()


# In[6]:


time = df.columns[58:77]

for region in df.index:
    for q in time:
        df.at[region,q] = df.at[region,q][0:6]
df


# In[7]:


data2 = df.reset_index(drop=True)

for c in data2.columns:
    data2[str(c)] = pd.to_numeric(data2[str(c)], errors='coerce')
    
data2.index = df.index
data2


# In[8]:


lst = []

dataFrame1 = pd.DataFrame(lst, index = df.index, columns = df.columns)
  
for region in data2.index:

        base = data2.loc[region:region,"2018Q1": "2018Q4"]
        baseAverage = base.mean(axis = 1)
        
        for quarter in data2.columns:
            CPI = data2.loc[region, quarter]/baseAverage[region]
            dataFrame1.at[region, quarter] = CPI
            
dataFrame1


# In[9]:


writer = pd.ExcelWriter('ThreeRoomProphet.xlsx')
dataFrame.to_excel(writer)
writer.save()


# In[ ]:




