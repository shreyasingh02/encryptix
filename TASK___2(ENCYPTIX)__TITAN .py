#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


titanic_data =pd.read_csv("Titanic-Dataset.csv")


# In[3]:


titanic_data.head()


# In[4]:


titanic_data.tail()


# In[5]:


titanic_data.shape


# In[6]:


titanic_data.info()


# In[7]:


titanic_data.isnull().sum()


# In[8]:


titanic_data = titanic_data.drop(columns='Cabin', axis=1)


# In[9]:


titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# In[10]:


print(titanic_data['Embarked'].mode())


# In[11]:


print(titanic_data['Embarked'].mode()[0])


# In[12]:


titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)


# In[13]:


titanic_data.isnull().sum()


# In[14]:


titanic_data.describe()


# In[15]:


titanic_data['Survived'].value_counts()


# In[16]:


sns.set()


# In[23]:


sns.countplot(x='Survived', data=titanic_data)
plt.show()


# In[24]:


titanic_data['Sex'].value_counts()


# In[25]:


sns.countplot(x='Sex', data=titanic_data)


# In[30]:


sns.countplot(x='Sex', hue='Survived', data=titanic_data)
plt.show()


# In[31]:


sns.countplot(x='Pclass', data=titanic_data)


# In[33]:


sns.countplot(x='Pclass', hue='Survived', data=titanic_data)


# In[34]:


titanic_data['Sex'].value_counts()


# In[35]:


titanic_data['Embarked'].value_counts()


# In[36]:



titanic_data.replace({'Sex':{'male':0,'female':1}, 'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)


# In[37]:


titanic_data.head()


# In[38]:


X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y = titanic_data['Survived']


# In[39]:


print(X)


# In[40]:


print(Y)


# In[41]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)


# In[42]:


print(X.shape, X_train.shape, X_test.shape)


# In[43]:


model = LogisticRegression()


# In[44]:


model.fit(X_train, Y_train)


# In[45]:


X_train_prediction = model.predict(X_train)


# In[46]:


print(X_train_prediction)


# In[47]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[48]:


X_test_prediction = model.predict(X_test)


# In[49]:


print(X_test_prediction)


# In[50]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# In[ ]:




