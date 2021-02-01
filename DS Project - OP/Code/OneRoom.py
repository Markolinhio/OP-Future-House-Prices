#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


get_ipython().system('pip install fbprophet')


# In[3]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from fbprophet import Prophet
from pandas import to_datetime
import datetime


# In[4]:


get_ipython().system('pip install plotly==4.11.0 ')


# In[5]:


df = pd.read_csv("Alldata2006SOneRoom.csv", sep = ';', encoding='cp437')
data = df.replace('...', np.nan).drop([ "Building type", "Number of rooms"], axis =1)
data.index = data["Region"]
data = data.drop(["Region"], axis = 1)
data = data.drop(['Uusimaa', 'Helsinki'])
data


# In[16]:


data2 = data.reset_index(drop=True)

for c in data2.columns:
    data2[str(c)] = pd.to_numeric(data2[str(c)], errors='coerce')
    
data2.index = data.index
data2


# In[17]:


#cells in the orignial data that differe +-450 from their neigbours
error = data2.diff(axis=1)
values1 = []
index = 0

for r in error.index:
    for q in error.columns:
        if abs(float(error.at[str(r), str(q)])) > 450:
            values1.append(tuple([str(r), str(q), error.at[str(r), str(q)]]))
            index = index + 1
print(index)
values1


# In[18]:


#regions that have less than 30% data
missing_values_count = data.isnull().sum(axis=1)
print(missing_values_count)
indexes3 = data.index.tolist()
deleted_rows = []
i = 0

      
for row in missing_values_count:
    if row > 0.7*58:
        deleted_rows.append(str(indexes3[i]))
    i = i + 1

print(deleted_rows)
print(str(len(deleted_rows)) + " regions have less than 30% of data filled one room flats")


# In[19]:


#how much data missing before imputations or deleting regions with less than 30% data
missing_values_count = data.isnull().sum(axis=1)

total_cells= np.product(data.shape)
total_missing = missing_values_count.sum()

(total_missing/total_cells)*100


# In[20]:


#delete regions that have less than 30%data
for i in deleted_rows:
    data = data.drop(str(i), axis = 0)

data


# In[21]:


#missing data after deleting the regions
missing_values_count = data.isnull().sum(axis=1)

total_cells= np.product(data.shape)
total_missing = missing_values_count.sum()

(total_missing/total_cells)*100


# In[22]:


#imputation with iterative imputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

imp = IterativeImputer(random_state=0)
final_data = imp.fit_transform(data.to_numpy())
final_data = pd.DataFrame(data = final_data, columns = data.columns)
final_data.index = data.index
final_data


# In[23]:


data1 = final_data.reset_index(drop=True)

for c in data1.columns:
    data1[str(c)] = pd.to_numeric(data1[str(c)], errors='coerce')
    
data1.index = final_data.index
data1


# In[35]:


writer = pd.ExcelWriter('Excel.xlsx')
data1.to_excel(writer)
writer.save()


# In[30]:


#recognizing cells that need to be manually altered
error = data1.diff(axis=1)
values = []
missing = []
hei = 0

for r in error.index:
    for q in error.columns:
        if abs(float(error.at[str(r), str(q)])) > 450:
            values.append(tuple([str(r), str(q), error.at[str(r), str(q)]])) 
for i in values:
    if i not in values1:
        missing.append(i)
        hei = hei + 1
       
missing


# In[32]:


timeseries = pd.DataFrame(final_data.iloc[0])
plt.rcParams.update({'figure.figsize': (50,15)})
plt.rcParams['xtick.labelsize'] = 25
plt.rcParams['ytick.labelsize'] = 25
plt.plot(timeseries)


# In[33]:


import datetime
def index_to_float(index):
    year = float(index[0:4])
    quarter = index[4:6]
    if quarter == 'Q1':
        return year + 0
    elif quarter == 'Q2':
        return year + 0.25
    elif quarter == 'Q3':
        return year + 0.5
    elif quarter == 'Q4':
        return year + 0.75
def float_to_index(index):
    year = int(index)
    quarter = index-year
    if quarter == 0:
        return str(year) + 'Q1'
    elif quarter == 0.25:
        return str(year) + 'Q2'
    elif quarter == 0.5:
        return str(year) + 'Q3'
    elif quarter == 0.75:
        return str(year) + 'Q4'

def quarter_to_date(index):
    year = index[0:4]
    quarter = index[4:6]
    if quarter == "Q1":
        return datetime.datetime(int(year), 3, 31)
    elif quarter == "Q2":
        return datetime.datetime(int(year), 6, 30)
    elif quarter == "Q3":
        return datetime.datetime(int(year), 9, 30)
    elif quarter == "Q4":
        return datetime.datetime(int(year), 12, 31)


# In[34]:


date_index = list(map(quarter_to_date, timeseries.index))
float_index = np.array(list(map(index_to_float, timeseries.index))).reshape(-1, 1)
i = 2020.25
temp = []
while i < 2030:
    i += 0.25
    temp.append(i)
predict_time = np.array(temp).reshape(-1, 1)
future_years = list(map(float_to_index, predict_time))


# In[35]:



def predict(index):
    timeseries = pd.DataFrame(final_data.iloc[index])
    #Linear Regression
    reg = LinearRegression().fit(float_index.reshape(-1, 1), timeseries.to_numpy())
    linear_pred = reg.predict(predict_time)
    linear_pred = pd.DataFrame(data = linear_pred, columns = timeseries.columns)
    linear_pred.index = future_years
    #Prophet
    model = Prophet()
    new_timeseries = pd.DataFrame(columns = ['ds', 'y'])
    new_timeseries['y'] = timeseries.iloc[:, 0]
    new_timeseries['ds'] = date_index
    model.fit(new_timeseries)
    future_timeseries = pd.DataFrame(columns = ['ds', 'y'])
    future_timeseries['ds'] = list(map(quarter_to_date, future_years))
    out_sample_forecast = model.predict(future_timeseries)
    prophet_pred = pd.DataFrame(out_sample_forecast['yhat'].to_numpy().flatten(), columns = timeseries.columns)
    prophet_pred.index = future_years
    prophet_pred_lower = pd.DataFrame(out_sample_forecast['yhat_lower'].to_numpy().flatten(), columns = timeseries.columns)
    prophet_pred_lower.index = future_years
    prophet_pred_upper = pd.DataFrame(out_sample_forecast['yhat_upper'].to_numpy().flatten(), columns = timeseries.columns)
    prophet_pred_upper.index = future_years
    #Visualize
    fig = go.Figure()
    fig.update_layout(plot_bgcolor = 'rgb(255,255,255)')
    present_time = timeseries.index.values
    future_time = np.append(['2020Q2*'], future_years)
    fig.add_trace(
        go.Scatter(
            x= present_time, 
            y=timeseries.values.flatten(), 
            name=timeseries.columns[0][0],
            line=dict(color='black', width=4)
            ))
    fig.add_trace(
        go.Scatter(
            x= future_time, 
            y = timeseries.tail(1).append(linear_pred).values.flatten(), 
            name = 'Linear', 
            line=dict(color='blue', width = 4)
            ))
    fig.add_trace(
        go.Scatter(
            x= future_time, 
            y = timeseries.tail(1).append(prophet_pred).values.flatten(), 
            name = 'Prophet', 
            line=dict(color='red', width = 4)
            ))
    fig.add_trace(
        go.Scatter(
            x= future_time, 
            y = timeseries.tail(1).append(prophet_pred_lower).values.flatten(), 
            name = 'Prophet_lower', 
            line=dict(color='gray', width = 2, dash='dash'),
            ))
    fig.add_trace(
        go.Scatter(
            x= future_time, 
            y = timeseries.tail(1).append(prophet_pred_upper).values.flatten(), 
            name = 'Prophet_upper', 
            line=dict(color='gray', width = 2, dash='dash'),
            ))
    return fig.show()
    #return (timeseries, linear_pred, poly_pred, prophet_pred)


# In[36]:


predict(0)


# In[37]:


def CPI(cities: list, years: list):
    
    lst = []
    
    dataFrame = pd.DataFrame(lst, index = cities, columns = years)
        
    for city in cities:
        base = final_data.loc[city:city,"2015Q1 Price per square meter (EUR/m2)": "2015Q4 Price per square meter (EUR/m2)"]
        baseAverage = base.mean(axis = 1)
        
        
        for quarter in years:
            CPI = final_data.loc[city, quarter + " Price per square meter (EUR/m2)"]/baseAverage[city]
                    
            dataFrame.at[city, quarter] = CPI
            
    return dataFrame


# In[38]:


moi = [ "Helsinki", "Kainuu", "Vantaa"]

hei = ["2016Q2", "2018Q1", "2018Q2", "2019Q3"]

CPI(moi, hei)


# In[39]:


def realPrice(cities: list, years: list):
    
    lst = []
    
    dataFrame = pd.DataFrame(lst, index = cities, columns = years)
        
    for city in cities:
        base = final_data.loc[city:city,"2015Q1 Price per square meter (EUR/m2)": "2015Q4 Price per square meter (EUR/m2)"]
        baseAverage = base.mean(axis = 1)
        
        
        for quarter in years:
            CPI = round(final_data.loc[city, quarter + " Price per square meter (EUR/m2)"]/baseAverage[city], 3)
            RPI = final_data.at[city,quarter + " Price per square meter (EUR/m2)"]/CPI
            dataFrame.at[city, quarter] = RPI
            
    return dataFrame
        


# In[41]:


df = pd.read_csv("LPOneRoom", sep = ';', encoding='cp437')
df.index = df["Region"]
df = df.drop(["Region"], axis = 1)
df


# In[43]:


writer = pd.ExcelWriter('ResultsOneRoomLinear.xlsx')
df.to_excel(writer)
writer.save()


# In[7]:


lst = []
    
dataFrame = pd.DataFrame(lst, index = df.index, columns = df.columns)
  
for region in df.index:

        base = df.loc[region:region,"2018Q1": "2018Q4"]
        baseAverage = base.mean(axis = 1)
        
        for quarter in df.columns:
            CPI = df.loc[region, quarter]/baseAverage[region]
            dataFrame.at[region, quarter] = CPI
            
dataFrame


# In[8]:


writer = pd.ExcelWriter('OneRoomLP.xlsx')
dataFrame.to_excel(writer)
writer.save()


# In[44]:


df = pd.read_csv("OneRoomProphet", sep = ';', encoding='cp437')
df.index = df["Region"]
df = df.drop(["Region"], axis = 1)
df


# In[45]:


writer = pd.ExcelWriter('ResultsOneRoomProphet.xlsx')
df.to_excel(writer)
writer.save()


# In[10]:


time = df.columns[58:77]

for region in df.index:
    for q in time:
        df.at[region,q] = df.at[region,q][0:6]
df


# In[11]:


data2 = df.reset_index(drop=True)

for c in data2.columns:
    data2[str(c)] = pd.to_numeric(data2[str(c)], errors='coerce')
    
data2.index = df.index
data2


# In[12]:


lst = []

dataFrame = pd.DataFrame(lst, index = df.index, columns = df.columns)
  
for region in data2.index:

        base = data2.loc[region:region,"2018Q1": "2018Q4"]
        baseAverage = base.mean(axis = 1)
        
        for quarter in data2.columns:
            CPI = data2.loc[region, quarter]/baseAverage[region]
            dataFrame.at[region, quarter] = CPI
            
dataFrame


# In[13]:


get_ipython().system('pip install openpyxl')


# In[14]:


writer = pd.ExcelWriter('OneRoomProphet.xlsx')
dataFrame.to_excel(writer)
writer.save()


# In[ ]:




