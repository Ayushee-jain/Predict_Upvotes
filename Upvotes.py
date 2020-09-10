#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data=pd.read_csv(r'C:\Users\LENOVO\Downloads\train_NIR5Yl1.csv')


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


data=data.drop(columns=['ID'])


# In[6]:


data.isnull().sum()


# In[7]:


data.corr()


# In[8]:


data=data.drop(columns=['Username'])


# In[9]:


test=pd.read_csv(r'C:\Users\LENOVO\Downloads\test_8i3B3FC.csv')
t=pd.read_csv(r'C:\Users\LENOVO\Downloads\test_8i3B3FC.csv')


# In[10]:


test.head()


# In[11]:


test.isnull().sum()


# In[12]:


test=test.drop(columns=['ID','Username'])


# In[13]:


df=pd.concat([data,test],axis=0)


# In[14]:


df.shape


# In[15]:


df.dtypes


# In[16]:


df['Tag'].unique()


# In[17]:


dummies=pd.get_dummies(df['Tag'],drop_first=True)


# In[18]:


dummies


# In[19]:


df=pd.concat([df,dummies],axis=1)


# In[20]:


df.shape


# In[21]:


df=df.drop(columns=['Tag'])


# In[22]:


train=df.iloc[:330045,:]
test=df.iloc[330045:,:]


# In[23]:


train.isnull().sum()


# In[24]:


test.isnull().sum()


# In[25]:


test=test.drop(columns=['Upvotes'])


# In[26]:


x=train.drop(columns=['Upvotes'])
y=train['Upvotes']


# In[27]:


y.shape


# In[28]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x=sc.fit_transform(x)
test=sc.fit_transform(test)
y=sc.fit_transform(y.values.reshape(-1,1))


# In[29]:


print(x)


# In[30]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3)


# In[31]:


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
regressor.fit(xtrain,ytrain)
y_pred=regressor.predict(xtest)


# In[32]:


from sklearn.metrics import mean_squared_error
print(mean_squared_error(ytest,y_pred))


# In[33]:


plt.scatter(ytest,y_pred)
plt.show()


# In[34]:


sns.distplot(ytest)


# In[35]:


sns.distplot(y_pred)


# In[36]:


y_pred=regressor.predict(test)


# In[37]:


y_pred=sc.inverse_transform(y_pred)


# In[38]:


y_pred


# In[39]:


sns.distplot(y_pred)


# In[43]:


submission=pd.DataFrame({'ID':t['ID'],'Upvotes':y_pred})


# In[44]:


submission.head()


# In[45]:


filename = 'Upvote Submission.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)


# In[ ]:




