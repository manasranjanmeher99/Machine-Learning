import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv(r"C:\Users\ASUS\Downloads\Salary_Data.csv")
print("Dataset Shape:",dataset.shape)

x= dataset.iloc[:,:-1]
y= dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


y_pred= regressor.predict(x_test)
print(y_pred)

comparison= pd.DataFrame({'Actual':y_test,'Predict':y_pred})
print(comparison)

plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train),color='blue')
plt.title('Salary vs Experiance(Test Set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()


model_coef= regressor.coef_
print(model_coef)

model_const= regressor.intercept_
print(model_const)


y_12= model_coef*12 +model_const
print(y_12)

y_20= model_coef*20 +model_const
print(y_20)

dataset.mean()        # this will give mean of that particular column

dataset['Salary'].mean()

from scipy.stats import variation
variation(dataset.values) # coefficience of entire dataframe

variation(dataset['Salary'])    # covariant of that entire data

# Correlation
dataset.corr()    # this will give correlation of entire dataframe

dataset['Salary'].corr(dataset['YearsExperience'])    # this will give us correlation bet these two data


# skewness
dataset.skew()  # give skewness of entire dataset
dataset['Salary'].skew()      # give skewness of particular column

# Standard Error
dataset.sem()  # give the standard error of entire dataframe
dataset['Salary'].sem()    # give the standard error of particular column

# Z-score
import scipy.stats as stats
dataset.apply(stats.zscore)    # give the z-score of entire dataframe
stats.zscore(dataset['Salary'])


#SSR
y_mean= np.mean(y)
ssr= np.sum((y_pred-y_mean)**2)
print(ssr)

#SSE
y = y[0:6]
sse = np.sum((y-y_pred)**2)
print(sse)


#SST
mean_total = np.mean(dataset.values)
sst= np.sum((dataset.values- mean_total)**2)
print(sst)

#r Squre
r_squre= 1-ssr/sst
print(r_squre)

bias= regressor.score(x_train, y_train)
print(bias)

variance= regressor.score(x_test, y_test)
print(variance)

import pickle

# save the training model to disk
filename= 'linear_regression_model.pkl'

#open a file in written in binary mode and dump the model
with open(filename, 'wb') as file:
    pickle.dump(regressor, file)

print("Model has been pickled and save linear_regression model.pkl")


import os 
os.getcwd()








