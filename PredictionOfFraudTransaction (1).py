#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[3]:


data=pd.read_csv('Fraud.csv.csv')


# In[4]:


data.head()


# In[5]:


data.tail()


# In[6]:


data.info()


# In[7]:


data.shape


# In[8]:


data.isna().sum()


# In[9]:


data.isFraud.value_counts()


# In[10]:


data.isFlaggedFraud.value_counts()


# In[12]:


data=data.drop(['nameOrig','nameDest'],axis=1)


# In[16]:


from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
data['type']=label_encoder.fit_transform(data['type'])


# In[18]:


x,y=data.loc[:,data.columns!='isFraud'],data['isFraud']


# In[43]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=42)


# In[44]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)


# In[45]:


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred=gnb.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# In[51]:


from sklearn.linear_model import LogisticRegression
Classifier=LogisticRegression(random_state=0)
Classifier.fit(x_train,y_train)
y_pred=Classifier.predict(x_test)
print("Accuracy:",metrics.accuracy_score(y_test,y_pred))


# In[ ]:


'''
Q2) Describe your fraud detection model in elaboration.

Certainly! The fraud detection model uses machine learning techniques to predict fraudulent transactions. Here's an overview of the key steps in the model development:

1. Data Loading:
The initial step involves loading a dataset containing information about financial transactions. The dataset is stored in a CSV file ('Fraud.csv.csv') and is read into a Pandas DataFrame named data.

2. Exploratory Data Analysis (EDA):
Exploratory Data Analysis is performed to understand the characteristics of the dataset:

head() and tail() show a glimpse of the first and last few rows.
info() provides information about data types and missing values.
shape indicates the number of rows and columns in the dataset.
isna().sum() reveals the count of missing values in each column.
value_counts() examines the distribution of values in the 'isFraud' and 'isFlaggedFraud' columns.
3. Data Preprocessing:
Unnecessary columns ('nameOrig' and 'nameDest') are dropped from the dataset.
The 'type' column, representing a categorical variable, is encoded using Label Encoding to convert it into a numerical format suitable for machine learning models.
4. Splitting Data:
The dataset is split into features (x) and the target variable (y). Then, a further split is performed to create training and testing sets using the train_test_split function from scikit-learn.

5. Feature Scaling:
Standard scaling is applied to normalize the feature values. This ensures that all features contribute equally to the machine learning models.

6. Model Building and Evaluation:
Gaussian Naive Bayes:
A Gaussian Naive Bayes model is trained using the GaussianNB class from scikit-learn. This model assumes that features are normally distributed. After training, predictions are made on the test set, and the accuracy of the model is evaluated using metrics.accuracy_score.

Logistic Regression:
A Logistic Regression model is trained using the LogisticRegression class. Logistic Regression is a linear model suitable for binary classification tasks. Similar to the Naive Bayes model, predictions are made, and accuracy is calculated.

Conclusion:
The script concludes by providing the accuracy of both the Gaussian Naive Bayes and Logistic Regression models on the test set. Accuracy is a common metric for classification problems, representing the proportion of correctly predicted instances. However, it's essential to consider other metrics and perform a more in-depth analysis based on the specific requirements and characteristics of the fraud detection problem, as accuracy alone may not be sufficient, especially in imbalanced datasets where fraudulent transactions are rare.
'''

'''
Q3) How did you select variables to be included in the model?

It's important to note that the specific approach to variable selection may vary based on the nature of the dataset, the problem at hand, and the characteristics of the features. Additionally, variable selection is often an iterative process, and the effectiveness of the chosen variables should be evaluated through model performance metrics and validation techniques.
'''
'''
Q5) What are the key factors that predict fraudulent customer?

Transaction Amount, Transaction Frequency, Time of Day or Week, User Behavior and Biometrics, IP Address and Geolocation, Device Information, Transaction Type and Patterns, Machine Learning Model Insights.
'''
'''
Q7) What kind of prevention should be adopted while company update its infrastructure?

Security Assessment, Regular Security Audits, Update and Patch Management, Network Segmentation, Data Encryption, User Authentication and Access Controls.
'''

'''
Q8) Assuming these actions have been implemented, how would you determine if they work?

Security Metrics and KPIs:

Define and regularly monitor security metrics and key performance indicators (KPIs) aligned with cybersecurity goals. Examples include the number of detected incidents, response times, and the success rate of security awareness programs.

Incident Detection and Response:

Assess the efficiency of incident detection and response mechanisms. Evaluate how well the organization identifies and mitigates security incidents, including the time taken to detect and respond to threats.

Security Audits and Assessments:

Regularly perform security audits and assessments to evaluate the overall security posture. This includes vulnerability assessments, penetration testing, and third-party security audits.
'''

