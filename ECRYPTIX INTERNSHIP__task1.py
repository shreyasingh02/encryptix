#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[3]:


from sklearn.metrics import accuracy_score,r2_score
from sklearn.metrics import classification_report


# In[4]:


df=pd.read_csv("IRIS.csv")


# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


df.shape


# In[9]:


df.describe()


# In[10]:


df.isnull().sum()


# In[12]:


sns.regplot(data=df,x=df['sepal_length'],y=df['sepal_width'],ci=None,marker="1",color="blue")
plt.show()


# In[13]:


sns.regplot(data=df,x=df['petal_length'],y=df['petal_width'],ci=None,marker="1",color="green")
plt.show()


# In[14]:


corr=df.corr()
sns.heatmap(corr,annot=True)


# In[15]:


sns.pairplot(df,hue='species')


# In[16]:


x=df[["sepal_length","sepal_width","petal_length","petal_width"]]
y=df["species"]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.2)


# In[17]:


sns.barplot(x='petal_width',y='petal_length' , data=df)
plt.xlabel ('petal_width')
plt.xlabel('petal_length')
plt. show()


# In[18]:


sns.barplot(x='sepal_width',y='sepal_length' , data=df)
plt.xlabel ('sepal_width')
plt.xlabel('sepal_length')
plt. show()


# In[19]:


sns.violinplot (x="species",y="petal_length", data=df, size=6)


# In[20]:


sns.violinplot (x="species",y="petal_width", data=df, size=6)


# In[21]:


sns.violinplot (x="species",y="sepal_width", data=df, size=6)


# In[23]:


sns.violinplot (x="species",y="sepal_length", data=df, size=6)


# In[24]:



sns.barplot(x='petal_length', y='species', data=df)
plt.xlabel('petal_length')
plt.ylabel('species')
plt.show()


# In[25]:


sns.barplot(x='petal_width', y='species', data=df)
plt.xlabel('petal_width')
plt.ylabel('species')
plt.show()


# In[26]:


sns.barplot(x='sepal_width', y='species', data=df)
plt.xlabel('sepal_width')
plt.ylabel('species')
plt.show()


# In[27]:


sns.barplot(x='sepal_length', y='species', data=df)
plt.xlabel('sepal_length')
plt.ylabel('species')
plt.show()


# In[28]:


sns.catplot(x='sepal_length', hue ='species', kind='count', data =df)


# In[29]:


sns.catplot(x='sepal_width', hue ='species', kind='count', data =df)


# In[30]:


sns.catplot(x='petal_length', hue ='species', kind='count', data =df)


# In[31]:


sns.catplot(x='petal_width', hue ='species', kind='count', data =df)


# In[32]:


df.shape


# In[33]:


print("x:",x_train.shape,x_test.shape)
print("y:",y_train.shape,y_test.shape)


# In[34]:


model=KNeighborsClassifier()
model.fit(x_train,y_train)


# In[35]:


y_pred=model.predict(x_test)
print(accuracy_score(y_test,y_pred))


# In[36]:


print(classification_report(y_test,y_pred))


# In[37]:


svm=SVC()
svm.fit(x_train,y_train)
pred=svm.predict(x_test)
print(accuracy_score(y_test,y_pred))


# In[ ]:




