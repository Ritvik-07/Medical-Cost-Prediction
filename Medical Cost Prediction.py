#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import re


# In[29]:


data = pd.read_csv('D:/mlp-course-material/MODULE 9 - Mini Projects/3. Medical Cost Prediction/medical_cost_data.csv')


# In[30]:


data.head(10)


# In[31]:


data.info()


# In[32]:


data.describe()


# In[33]:


data.isnull().sum()


# In[36]:


data.smoker.unique()


# In[38]:


data.region.unique()


# In[37]:


data.sex.unique()


# In[42]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data.sex.drop_duplicates())
data.sex = le.transform(data.sex)


le.fit(data.smoker.drop_duplicates())
data.smoker = le.transform(data.smoker)


le.fit(data.region.drop_duplicates())
data.region = le.transform(data.region)


# In[43]:


data.head()


# In[45]:


data.corr()['charges'].sort_values()


# In[49]:


heatmap = sns.heatmap(data[['region', 'sex', 'children','bmi','age', 'smoker', 'charges']].corr(),annot = True)


# In[55]:


sns.catplot(x = 'smoker', kind = 'count', hue = 'sex', height = 5 ,data = data)


# In[58]:


ax = sns.displot(data['bmi'])


# In[64]:


sns.catplot(x = 'children', kind = 'count', data = data, height= 5)


# In[65]:


data.head()


# In[68]:


x = data.drop(data.columns[[6, 5]], axis = 1)
y = data['charges']


# In[71]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.30, random_state = 0)


# In[72]:


from sklearn.preprocessing import MinMaxScaler


# In[73]:


scaler = MinMaxScaler()

x = scaler.fit_transform(x)


# In[81]:


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
from sklearn.linear_model import LinearRegression


# In[82]:


dt =  DecisionTreeRegressor()
rf = RandomForestRegressor()
linear = LinearRegression()
svr = svm.SVR()


# In[83]:


linear.fit(xtrain, ytrain)


# In[84]:


svr.fit(xtrain, ytrain)


# In[85]:


rf.fit(xtrain, ytrain)


# In[86]:


dt.fit(xtrain, ytrain)


# In[88]:


y_pred_linear = linear.predict(xtest)
y_pred_dt = dt.predict(xtest)
y_pred_rf = rf.predict(xtest)
y_pred_svr = svr.predict(xtest)


# In[89]:


from sklearn.metrics import mean_squared_error


# In[92]:


import math
error_linear = math.sqrt(mean_squared_error((y_pred_linear), ytest))
error_dt = math.sqrt(mean_squared_error((y_pred_dt), ytest))
error_rf = math.sqrt(mean_squared_error((y_pred_rf), ytest))
error_svr = math.sqrt(mean_squared_error((y_pred_svr), ytest))


# In[93]:


print('Model                               :     RMSE error\n')
print('LinearRegressor               : ', error_linear)
print('DecisionTreeRegressor    : ', error_dt)
print('RandomForestRegressor  : ', error_rf)
print('SVR                                  : ', error_svr)


# In[ ]:




