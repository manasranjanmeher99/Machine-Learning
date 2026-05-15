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
svr_reg= SVR(kernel='sigmoid', degree=4, C=10.0, gamma= 'auto')
svr_reg.fit(x,y)

svr_model_pred= svr_reg.predict([[6.5]])
print(svr_model_pred)

#svr model
from sklearn.svm import SVR
svr_reg= SVR(kernel='poly', degree=4, C=1.0, gamma='auto')
svr_reg.fit(x,y)

svr_model_pred= svr_reg.predict([[6.5]])
print(svr_model_pred)

#svr model
from sklearn.svm import SVR
svr_reg= SVR(kernel='linear', degree=4, C=1.0, gamma='auto')
svr_reg.fit(x,y)

svr_model_pred= svr_reg.predict([[6.5]])
print(svr_model_pred)

#svr model
from sklearn.svm import SVR
svr_reg= SVR(kernel='poly', degree=4, C=1.0, gamma='auto')
svr_reg.fit(x,y)

svr_model_pred= svr_reg.predict([[11]])
print(svr_model_pred)