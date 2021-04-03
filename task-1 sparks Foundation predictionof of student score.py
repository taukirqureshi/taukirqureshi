#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns 

# To ignore the warnings 
import warnings as wg
wg.filterwarnings("ignore")


# In[2]:


url = "https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
df = pd.read_csv(url)


# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum()


# In[9]:


plt.rcParams["figure.figsize"] = [16,9]
df.plot(x='Hours', y='Scores', style='*', color='orange', markersize=10)
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.grid()
plt.show()


# In[10]:


df.head()


# In[11]:


X = df.iloc[:, :1].values  
y = df.iloc[:, 1:].values


# In[34]:


X


# In[17]:


y


# In[18]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# In[19]:


from sklearn.linear_model import LinearRegression  

model = LinearRegression()  
model.fit(X_train, y_train)


# In[20]:


line = model.coef_*X + model.intercept_

# Plotting for the training data
plt.rcParams["figure.figsize"] = [16,9]
plt.scatter(X_train, y_train, color='purple')
plt.plot(X, line, color='green');
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.grid()
plt.show()


# In[33]:


# Plotting for the testing data
plt.rcParams["figure.figsize"] = [16,9]
plt.scatter(X_test, y_test, color='red')
plt.plot(X, line, color='green');
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.grid()
plt.show()


# In[27]:


print(X_test) # Testing data - In Hours
y_pred = model.predict(X_test) # Predicting the scores


# In[28]:


# Comparing Actual vs Predicted

y_test


# In[29]:


y_pred


# In[30]:


# Comparing Actual vs Predicted
comp = pd.DataFrame({ 'Actual':[y_test],'Predicted':[y_pred] })
comp


# In[31]:


# Testing with your own data

hours = 9.25
own_pred = model.predict([[hours]])
print("The predicted score if a person studies for",hours,"hours is",own_pred[0])


# In[32]:


from sklearn import metrics  

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




