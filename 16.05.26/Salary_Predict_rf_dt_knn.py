import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset= pd.read_csv(r"C:\Users\ASUS\Downloads\emp_sal.csv")

x= dataset.iloc[:,1:2].values
y= dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg= LinearRegression()
lin_reg.fit(x,y)

# Linear regression
plt.scatter(x, y, color= 'red')
plt.plot(x,lin_reg.predict(x), color='blue')
plt.title('Linear Regression Graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_model_pred= lin_reg.predict([[6.5]])
print(lin_model_pred)

# Polynomial Model- degree 5
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=5)
x_poly= poly_reg.fit_transform(x)

poly_reg.fit(x_poly,y)
lin_reg_2= LinearRegression()
lin_reg_2.fit(x_poly, y)

plt.scatter(x, y, color= 'red')
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)), color='blue')
plt.title('Truth or Bluff(Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred


#svr model
from sklearn.svm import SVR
svr_reg= SVR(kernel='poly', degree=4, C=1.0, gamma='auto')
svr_reg.fit(x,y)

svr_model_pred= svr_reg.predict([[6.5]])
print(svr_model_pred)



# KNN weights="distance"
from sklearn.neighbors import KNeighborsRegressor
knn_reg= KNeighborsRegressor(weights="distance")
knn_reg.fit(x,y)

knn_reg_pred= knn_reg.predict([[6.5]])
print(knn_reg_pred)



# Decission Tree Regressor 
# (criterion="poisson", max_depth=5, min_samples_split=3)

from sklearn.tree import DecisionTreeRegressor
dt_reg= DecisionTreeRegressor(criterion="poisson", max_depth=5, min_samples_split=3)
dt_reg.fit(x,y)

dt_reg_pred= dt_reg.predict([[6.5]])
print(dt_reg_pred)



# random forest
from sklearn.ensemble import RandomForestRegressor
rf_reg= RandomForestRegressor(random_state=0, n_estimators=15, min_samples_split=3)
rf_reg.fit(x,y)

rf_reg_pred= rf_reg.predict([[6.5]])
print(rf_reg_pred)



