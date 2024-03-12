#!/usr/bin/env python
# coding: utf-8

# <h1 align="center" style="color:#E80F88;">Inventory Demand Forecasting</h1>

# #### Inventory forecasting is a method used to predict inventory levels for a future time period. It also helps keep track of sales and demand so you can better manage your purchase orders. In this , we will try to implement a machine learning model which can predict the sales for the different products which are sold in different stores.

# By accurately predicting sales, businesses can adjust their inventory levels accordingly to meet customer demand while minimizing excess inventory or stockouts.

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import warnings
import datetime
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from xgboost import XGBRegressor 

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error


# <h1 style="color:#982176;">Data Collection</h1>

# In[3]:


df = pd.read_csv('SuperStore_Sales_Dataset.csv')
df


# In[3]:


df.head()


# In[50]:


#First I will perform a preliminary analysis to understand the structure and types of data columns


# In[4]:


df.shape


# In[5]:


df.size


# In[6]:


df.info()


# In[7]:


df.describe()


# In[51]:


df['Category'].nunique()


# In[52]:


df['Category'].value_counts()    #Occurances of each unique value


# In[53]:


df['Country'].value_counts()


# In[54]:


df['Sub-Category'].value_counts()


# In[4]:


df['Sales'].nunique()


# In[55]:


df['Sales'].value_counts()


# <h1 style="color:#982176;">Data Cleaning</h1>

# In[8]:


#Checking if there are any missing value


# In[5]:


df.isnull()


# In[9]:


df.isnull().sum()


# In[6]:


df.duplicated()


# In[3]:


df.duplicated().sum()


# <h1 style="color:#982176;">Checking Outliers</h1>

# In[4]:


plt.figure(figsize=(14,10))
sns.set_style(style='whitegrid')
plt.subplot(2,2,1)
sns.boxplot(x='Row ID+O6G3A1:R6',data=df)
plt.subplot(2,2,2)
sns.boxplot(x='Sales',data=df)
plt.subplot(2,2,3)
sns.boxplot(x='Quantity',data=df)
plt.subplot(2,2,4)
sns.boxplot(x='Profit',data=df)


# In[5]:


# Function to remove outliers from a DataFrame column
def remove_outliers(df, column):
    mean = df[column].mean()
    std = df[column].std()
    upper_bound = mean + 3 * std
    lower_bound = mean - 3 * std
    outliers = df[(df[column] > upper_bound) | (df[column] < lower_bound)]
    df = df[(df[column] <= upper_bound) & (df[column] >= lower_bound)]
    return df, len(outliers)

# Columns for outlier removal
columns = ['Sales', 'Quantity', 'Profit']

# Loop through columns to remove outliers
for column in columns:
    print(f'Before removing outliers for {column}:', len(df))
    df, outliers_count = remove_outliers(df, column)
    print(f'Outliers for {column}:', outliers_count)
    print(f'After removing outliers for {column}:', len(df))


# <h1 style="color:#982176;">Exploratory Data Analysis(EDA)</h1>

# In[10]:


pip install wordcloud


# In[11]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud

plt.subplots(figsize=(20, 8))
wordcloud = WordCloud(background_color='white', width=1920, height=1080).generate(" ".join(df['Category']))

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.savefig('cast.png')
plt.show()


# In[6]:


sns.scatterplot(data=df, x='Sales', y='Profit',hue='Segment',style='Country')


# In[7]:


plt.figure(figsize=(14,10))
sns.set_style(style='whitegrid')
plt.subplot(2,2,1)
sns.kdeplot(x='Sales',data=df)
plt.subplot(2,2,2)
sns.kdeplot(x='Quantity',data=df)
plt.subplot(2,2,3)
sns.kdeplot(x='Profit',data=df)
plt.subplot(2,2,4)
sns.kdeplot(x='Returns',data=df)


# In[8]:


category_sales = df.groupby('Category')['Sales'].sum()

plt.figure(figsize=(6, 6))
colors = ['#DA0C81','#DFCCFB','#8E8FFA']
plt.pie(category_sales, labels=category_sales.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Percentage of Sales by Category',fontweight='bold')
plt.axis('equal') 
plt.show()


# In[9]:


df.sort_values(by='Order Date', inplace=True)

plt.figure(figsize=(10, 6))
plt.plot(df['Order Date'], df['Profit'], color='#319DA0', marker='o', linestyle='-')
plt.xlabel('Order Date')
plt.ylabel('Profit')
plt.title('Profit Over Time',fontweight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[10]:


plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Sub-Category", y="Quantity",data=df)
plt.title("Sub-Category vs Quantity",fontsize=15,fontweight='bold')
plt.xlabel("Sub-Category")
plt.ylabel("Quantity")
plt.xticks(rotation=45, ha='right')
plt.show()


# In[11]:


plt.style.use("default")
plt.figure(figsize=(5,5))
sns.barplot(x="Sub-Category", y="Sales", data=df)
plt.title("Sub-Category vs Sales",fontsize=15,fontweight='bold')
plt.xlabel("Sub-Category ")
plt.ylabel("Sales")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels(horizontal alignment)
plt.show()


# ### checking the variation of stock

# In[12]:


plt.figure(figsize=(10,5)) 
df.groupby('Order Date').mean()['Sales'].plot() 
plt.show()


# In[13]:


df.corr()


# In[14]:


corr_matrix = df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
plt.title('Correlation Heatmap')
plt.show()


# In[15]:


#lets find the categorialfeatures
list_1=list(df.columns)


# In[16]:


list_cate=[]
for i in list_1:
    if df[i].dtype=='object':
        list_cate.append(i)


# In[17]:


le=LabelEncoder()


# In[18]:


for i in list_cate:
    df[i]=le.fit_transform(df[i])


# In[19]:


df


# <h1 style="color:#982176;">Feature Engineering</h1>

# In[20]:


X = df.drop(columns=['Sales'])  # Features
y = df['Sales']  # Target variable


# In[21]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[22]:


# Linear Regression
print("Linear Regression:")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
print("Mean Squared Error (Linear Regression):", mse_lr)
print("R^2 Score (Linear Regression):", r2_lr)

# Random Forest Regression
print("\nRandom Forest Regression:")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
print("Mean Squared Error (Random Forest Regression):", mse_rf)
print("R^2 Score (Random Forest Regression):", r2_rf)

# Gradient Boosting Regression
print("\nGradient Boosting Regression:")
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
print("Mean Squared Error (Gradient Boosting Regression):", mse_gb)
print("R^2 Score (Gradient Boosting Regression):", r2_gb)

# XGBoost Regression
print("\nXGBoost Regression:")
xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
print("Mean Squared Error (XGBoost Regression):", mse_xgb)
print("R^2 Score (XGBoost Regression):", r2_xgb)


# <h1 style="color:#982176;">Model Evaluation</h1>

# In[28]:


# List of algorithms and their corresponding MSE and R^2 score values
algorithms = ['Linear Regression', 'Random Forest Regression', 'Gradient Boosting Regression', 'XGBoost Regression']
mse_values = [49864.23156550106, 22741.110094060157, 24052.546091053173, 21235.371908464793]
r2_scores = [0.2246636713262662, 0.6463996685252715, 0.6260082187971536, 0.6698123128218112]

# Plot the line chart for MSE
plt.figure(figsize=(10, 6))
plt.plot(algorithms, mse_values, marker='o', linestyle='-', color='b', label='MSE')
plt.xlabel('Algorithm')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Comparison of Mean Squared Error (MSE) and R^2 Score for Regression Algorithms')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.7)

# Annotate each point with its MSE value
for i, mse in enumerate(mse_values):
    plt.annotate(f'{mse:.2f}', (algorithms[i], mse), textcoords="offset points", xytext=(0,10), ha='center')

# Create a second y-axis for R^2 score
plt.twinx()
plt.plot(algorithms, r2_scores, marker='s', linestyle='-', color='r', label='R^2 Score')
plt.ylabel('R^2 Score')
plt.xticks(rotation=45)

# Annotate each point with its R^2 score
for i, r2 in enumerate(r2_scores):
    plt.annotate(f'{r2:.2f}', (algorithms[i], r2), textcoords="offset points", xytext=(0,10), ha='center', color='r')

# Show plot
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


# <h1 style="color:#982176;">Model Deployment</h1>

# In[30]:


print("Mean Absolute Error (XGBoost Regression):", mse_xgb)
print("Root Mean Squared Error (XGBoost Regression):", r2_xgb)

# Plot actual vs. predicted sales
plt.plot(y_test.values, label='Actual Sale', color='orange')
plt.plot(y_pred_xgb, label='Predicted Sale', color='green')
plt.xlabel('Data Point')
plt.ylabel('Sales')
plt.title('Actual vs. Predicted Sales (XGBoost Regression)')
plt.legend()
plt.show()


# In[32]:


actual_sales = y_test.values
predicted_sales = y_pred_xgb

# Scatter plot for actual vs. predicted sales
plt.figure(figsize=(8, 6))
plt.scatter(range(len(actual_sales)), actual_sales, label='Actual Sale', color='orange', alpha=0.7)
plt.scatter(range(len(predicted_sales)), predicted_sales, label='Predicted Sale', color='green', alpha=0.7)
plt.xlabel('Data Point')
plt.ylabel('Sales')
plt.title('Actual vs. Predicted Sales (XGBoost Regression)')
plt.legend()
plt.show()


# In[ ]:




