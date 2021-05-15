#!/usr/bin/env python
# coding: utf-8

# ## Real Estate - Price Predictor

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("dataset.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing.describe()


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


#for plotting histogram
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))


# ## Train-Test Splitting

# In[9]:


#for learning purpose
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[10]:


#train_set, test_set = split_train_test(housing, 0.2)


# In[11]:


#print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[12]:


from sklearn.model_selection import train_test_split 
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[14]:


strat_test_set.info()


# In[15]:


# 95/7 = 376/28


# In[16]:


housing = strat_train_set.copy()


# ## Looking for Correlations

# In[17]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[18]:


from pandas.plotting import scatter_matrix
attributes = ["RM", "ZN", "MEDV", "LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8))


# In[19]:


housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)


# ## Trying out attribute combinations 

# In[20]:


housing["TAXRM"] = housing['TAX']/housing['RM']


# In[21]:


housing.head()


# In[22]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[23]:


housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)


# In[24]:


housing = strat_train_set.drop("MEDV", axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# In[25]:


median =  housing["RM"].median()


# In[26]:


housing["RM"].fillna(median)


# In[27]:


housing.shape


# In[28]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[29]:


imputer.statistics_.shape


# In[30]:


X = imputer.transform(housing)


# In[31]:


housing_tr = pd.DataFrame(X, columns=housing.columns)


# In[32]:


housing_tr.describe()


# ## Scikit-learn Design

# Primarily, three types of objects
# 1. Estimators - estimates some parameter eg. Imputer
# fit method -  fits the dataset
# 
# 2. Transformers - tranform method takes input and returns the output based on the leanring from fit(). it also has a convenience function called fit_tranform()
# 
# 3. Predictors - linearregression model is an example of predictor. fit() and predic() are two common func.
# score() evaluate the predictors.

# ## feature scalling

# two types of feature scalling methods:
# 1. min-max scalling(normalisation)
#     (value-min/max-min) ----> 0 to 1
#     sklearn provides a class called MinMaxScaler 
# 
# 2. Standardization 
#     (value-mean)/std
#     sklearn provides a class called StandardScaler 

# ## Creating a Pipeline

# In[33]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #...... add as many as you want
    ('std_scaler', StandardScaler()),
])


# In[34]:


housing_num_tr =  my_pipeline.fit_transform(housing_tr)


# In[35]:


housing_num_tr


# ## selecting a desired model for real estates

# In[36]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model =  LinearRegression()
model = RandomForestRegressor()
#model = DecisionTreeRegressor()
model.fit(housing_num_tr, housing_labels)


# In[37]:


some_data = housing.iloc[:5]


# In[38]:


some_labels = housing_labels.iloc[:5]


# In[39]:


prepared_data = my_pipeline.transform(some_data)


# In[40]:


model.predict(prepared_data)


# In[41]:


list(some_labels)


# ## evaluating the model

# In[42]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[43]:


mse


# ## using better evaluattion technique - cross validation

# In[44]:


# 1 2 3 4 5 6 7 8 9 10
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[45]:


rmse_scores


# In[46]:


def print_scores(scores):
        print("Scores:", scores)
        print("Mean: ", scores.mean())
        print("Standard deviation: ", scores.std())


# In[47]:


print_scores(rmse_scores)


# ## saving the model

# In[48]:


from joblib import dump, load
dump(model, 'realestate.joblib')


# ## testing the model

# In[49]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_predictions, list(Y_test))


# In[50]:


final_rmse


# In[52]:


prepared_data[0]


# In[ ]:




