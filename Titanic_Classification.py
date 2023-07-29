#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data = pd.read_csv('Titanic-Dataset.csv')


# In[3]:


data.head()


# In[4]:


def preprocess_data(df):
    df['Age'].fillna(df['Age'].median(), inplace=True)

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

    df = pd.get_dummies(df, columns=['Embarked'], prefix='Embarked', drop_first=True)

    return df

data = preprocess_data(data)


# In[5]:


X = data[['Pclass', 'Age', 'Sex', 'Embarked_Q', 'Embarked_S']]
y = data['Survived']


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}


# In[8]:


rf_model = RandomForestClassifier(random_state=42)


# In[9]:


grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)


# In[10]:


best_rf_model = grid_search.best_estimator_


# In[11]:


predictions = best_rf_model.predict(X_test)


# In[12]:


accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy : {accuracy:.2f}")


# In[13]:


print("Best Hyperparameters:")
print(grid_search.best_params_)


# In[14]:


print("Classification Report:")
print(classification_report(y_test, predictions))


# In[15]:


conf_matrix = confusion_matrix(y_test, predictions)
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()


# In[16]:


feature_importances = dict(zip(X.columns, best_rf_model.feature_importances_))
print("Feature Importances:")
for feature, importance in feature_importances.items():
    print(f"{feature}: {importance:.4f}")


# In[17]:


rf_model.fit(X,y)


# In[18]:


def predict_survival(person_info):
    person_data = pd.DataFrame([person_info], columns=['Pclass', 'Age', 'Sex'])
    person_data['Sex'] = person_data['Sex'].map({'male': 0, 'female': 1})
    person_data['Embarked_Q'] = 0
    person_data['Embarked_S'] = 0

    survival_prediction = rf_model.predict(person_data)[0]
    return 'Survived' if survival_prediction == 1 else 'Not Survived'


# In[19]:


person_info = {
    'Pclass': 3,
    'Age': 4,
    'Sex': 'female'
}


# In[20]:


result = predict_survival(person_info)
print("Prediction:", result)


# In[ ]:




