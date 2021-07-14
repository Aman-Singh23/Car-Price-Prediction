#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction

# ## Problem Statement :
#     To predict the price of used car.

# ## Task :
#     Building model for predicting the price of used cars using multiples machine learning regression technique.

# ## Description
#     This dataset contains information about used cars.
#     This data can be used for a lot of purposes such as price prediction to exemplify the use of regression technique in Machine Learning.
#     The columns in the given dataset are as follows:
# 
#     1. name
#     2. year
#     3. selling_price
#     4. km_driven
#     5. fuel
#     6. seller_type
#     7. transmission
#     8. Owner
#     
#     Dataset Link : https://www.kaggle.com/nehalbirla/vehicle-dataset-from-cardekho

# ## Steps

# In[1]:


from IPython.display import Image
Image(r"C:\Users\Ninja Clasher\Desktop\Steps.jpg")


# ##### Importing Libraries

# In[2]:


import numpy as np           # used for advanced mathematical opertion.
import pandas as pd          # used for analysing and handling data.


# In[3]:


# for visualization
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


# for ignoring warning
from warnings import filterwarnings
filterwarnings('ignore')


# ##### Importing Data

# In[5]:


df = pd.read_csv(r"C:\Users\Ninja Clasher\Desktop\Car Price Prediction\Dataset\Car details.csv")


# In[6]:


df.head()


# In[7]:


df.drop(['Unnamed: 0'] , axis = 1 ,inplace = True)


# In[8]:


df.shape         # shape of dataset (i.e... 8128 records with 8 features columns)


# ##### Analysing Categorical Columns

# In[9]:


print(df['fuel'].unique())
print(df['seller_type'].unique())
print(df['transmission'].unique())
print(df['owner'].unique())


# ## EDA (Exploratory Data Analysis)

# In[10]:


df.isnull().sum()


# In[11]:


df.describe()


# In[12]:


df.columns


# In[13]:


final_dataset = df[['year', 'selling_price', 'km_driven', 'fuel', 'seller_type',
       'transmission', 'owner']]


# In[14]:


final_dataset.head()


# In[15]:


final_dataset['Current_Year']=2021
final_dataset['no_year'] = final_dataset['Current_Year']-final_dataset['year']


# In[16]:


final_dataset.drop(['year'] , axis = 1 ,inplace = True)
final_dataset.drop(['Current_Year'] , axis = 1 ,inplace = True)


# In[17]:


final_dataset.head()


# In[18]:


final_dataset.info()


# ### Pandas Profiling

# In[19]:


from pandas_profiling import ProfileReport
profile = ProfileReport(final_dataset)
profile.to_file(output_file='cars.html')


# ## Pre-processing
# 
# ##### Coverting categorical features into numerical using Dummy

# In[20]:


final_dataset = pd.get_dummies(final_dataset , drop_first= True)


# In[21]:


final_dataset.head()


# ### Correlation

# In[22]:


final_dataset.corr()


# In[23]:


sns.pairplot(final_dataset)


# In[24]:


corrmat = final_dataset.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))

g = sns.heatmap(final_dataset[top_corr_features].corr() , annot=True , cmap='Blues')


# In[25]:


final_dataset.head()


# ### Assigning X and Y

# In[26]:


X = final_dataset.iloc[:,1:]
y = final_dataset.iloc[:,0]


# ### Train Test Split

# In[27]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 , random_state=0)


# In[28]:


print(X_train.shape)
print(y_train.shape)

print(X_test.shape)
print(y_test.shape)


# ### Building Model

# In[29]:


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()


# ### Randomized SearchCV

# In[30]:


'''
from sklearn.model_selection import RandomizedSearchCV

random_grid = {'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200],
               'max_features': ['auto', 'sqrt'],
               'max_depth': [5, 10, 15, 20, 25, 30],
               'min_samples_split': [2, 5, 10, 15, 100],
               'min_samples_leaf': [1, 2, 5, 10]
              }

rf_random = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid,scoring='neg_mean_squared_error',
                               n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = 1)

rf_random.fit(X_train,y_train)
'''


# In[31]:


# print(rf_random.best_params_)


# In[32]:


# print(rf_random.best_score_)


# ### Best parameter we get for Random Forest Regressor
#     n_estimators : 300
#     min_samples_split : 15
#     min_samples_leaf : 1
#     max_features : 'sqrt'
#     max_depth : 15

# ### Tuned Random Forest Regressor

# In[33]:


from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators= 300, min_samples_split= 15,
                                min_samples_leaf= 1, max_features= 'sqrt', max_depth= 15,random_state=42)
regressor.fit(X_train,y_train)
pred = regressor.predict(X_test)


# In[34]:


print(pred)


# In[35]:


sns.distplot(y_test-pred)


# In[36]:


plt.scatter(y_test,pred)


# ### Measuring Metrics

# In[37]:


from sklearn.metrics import r2_score,mean_squared_error , mean_absolute_error


# In[38]:


r2=r2_score(y_test,pred)
print("R-squared:",r2)

rmse=np.sqrt(mean_squared_error(y_test,pred))
print("RMSE:",rmse)

adjusted_r_squared = 1 - (1-r2)*(len(y)-1)/(len(y)-X.shape[1]-1)
print("Adj R-square:",adjusted_r_squared)


# ### Saving Model

# In[40]:


import pickle
# open a file, where you ant to store the data
file = open('rf_reg_model.pkl', 'wb')

# dump information to that file
pickle.dump(regressor, file)


# In[ ]:




