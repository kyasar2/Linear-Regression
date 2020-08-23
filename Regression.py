#!/usr/bin/env python
# coding: utf-8



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# 
# We'll work with the Ecommerce Customers csv file from the company. It has Customer info, suchas Email, Address, and their color Avatar. Then it also has numerical value columns:
# 
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 
# 
# ** Read in the Ecommerce Customers csv file as a DataFrame called customers.**

# In[2]:


df = pd.read_csv("Ecommerce Customers")


# **Check the head of customers, and check out its info() and describe() methods.**

# In[8]:


df.head()


# In[6]:


df.describe()


# In[7]:


df.info()


# ## Exploratory Data Analysis
# 
# **Let's explore the data!**
# 
# For the rest of the exercise we'll only be using the numerical data of the csv file.
# ___
# **Use seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns. Does the correlation make sense?**

# In[10]:


sns.jointplot('Time on Website', 'Yearly Amount Spent', data = df )


# In[281]:





# ** Do the same but with the Time on App column instead. **

# In[11]:


sns.jointplot('Time on App', 'Yearly Amount Spent', data = df )


# ** Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership.**

# In[283]:





# **Let's explore these types of relationships across the entire data set. Use [pairplot](https://stanford.edu/~mwaskom/software/seaborn/tutorial/axis_grids.html#plotting-pairwise-relationships-with-pairgrid-and-pairplot) to recreate the plot below.(Don't worry about the the colors)**

# In[12]:


sns.pairplot(df)


# **Based off this plot what looks to be the most correlated feature with Yearly Amount Spent?**

# In[285]:


Length of Membership


# **Create a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership. **

# In[14]:


sns.lmplot( 'Yearly Amount Spent', 'Length of Membership', df)


# ## Training and Testing Data
# 
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# ** Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column. **

# In[18]:


X = df[['Avg. Session Length', 'Length of Membership', 'Time on App', 'Time on Website']]


# In[19]:


y = df['Yearly Amount Spent']


# ** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**

# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.3, random_state = 101 )


# ## Training the Model
# 
# Now its time to train our model on our training data!
# 
# ** Import LinearRegression from sklearn.linear_model **

# In[22]:


from sklearn.linear_model import LinearRegression


# **Create an instance of a LinearRegression() model named lm.**

# In[24]:


lm = LinearRegression()


# ** Train/fit lm on the training data.**

# In[25]:


lm.fit(X_train, Y_train )


# **Print out the coefficients of the model**

# In[29]:


print( lm.coef_ )


# ## Predicting Test Data
# Now that we have fit our model, let's evaluate its performance by predicting off the test values!
# 
# ** Use lm.predict() to predict off the X_test set of the data.**

# In[30]:


Y_pred = lm.predict(X_test )


# ** Create a scatterplot of the real test values versus the predicted values. **

# In[31]:


sns.scatterplot(Y_test, Y_pred)


# ## Evaluating the Model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**

# In[33]:


from sklearn import metrics 
mae = metrics.mean_absolute_error(Y_test, Y_pred )
mse = metrics.mean_squared_error(Y_test, Y_pred )


# ## Residuals
# 
# You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data. 
# 
# **Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().**

# In[35]:


sns.distplot(Y_test- Y_pred, bins = 40 )


# ## Conclusion
# We still want to figure out the answer to the original question, do we focus our efforst on mobile app or website development? Or maybe that doesn't even really matter, and Membership Time is what is really important.  Let's see if we can interpret the coefficients at all to get an idea.
# 
# ** Recreate the dataframe below. **

# In[298]:



