#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


credit_card_data = pd.read_csv('creditcard.csv')


# In[3]:


credit_card_data.head()


# In[4]:


credit_card_data.tail()


# In[5]:


credit_card_data.info()


# In[6]:


credit_card_data.isnull().sum()


# In[ ]:


##This Dataset is highly unblanced

0 --> Normal Transaction

1 --> fraudulent transaction


# In[7]:


credit_card_data['Class'].value_counts()


# In[8]:


legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[9]:


print(legit.shape)
print(fraud.shape)


# In[10]:


legit.Amount.describe()


# In[11]:


fraud.Amount.describe()


# In[12]:


credit_card_data.groupby('Class').mean()


# In[13]:


legit_sample = legit.sample(n=492)


# In[14]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)


# In[15]:


new_dataset.head()


# In[16]:


new_dataset.tail()


# In[17]:


new_dataset['Class'].value_counts()


# In[18]:


new_dataset.groupby('Class').mean()


# In[19]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[20]:


print(X)


# In[21]:


print(Y)


# In[22]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)


# In[23]:


print(X.shape, X_train.shape, X_test.shape)


# In[24]:


model = LogisticRegression()


# In[25]:


model.fit(X_train, Y_train)


# In[ ]:




