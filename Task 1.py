#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[54]:


data=pd.read_csv('titanic (2).csv')


# In[55]:


data.head()


# In[56]:


data.info()


# In[59]:


# Fill missing 'Age' values with the median age
imputer = SimpleImputer(strategy='median')
data['age'] = imputer.fit_transform(data[['age']])

# Fill missing 'Embarked' values with the most frequent value
data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)


# In[77]:


# Convert 'Sex' to numerical values
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])


# In[79]:


X = data[['pclass','sex','age','sibsp','parch', 'fare']]
y = data['survived']


# In[80]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[81]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[ ]:





# In[ ]:




