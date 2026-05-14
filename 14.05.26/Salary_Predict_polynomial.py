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


#Polynomial Model
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures()
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

# Polynomial Model- degree 3
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=3)
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

# Polynomial Model- degree 4
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=4)
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


# Polynomial Model- degree 6
from sklearn.preprocessing import PolynomialFeatures
poly_reg= PolynomialFeatures(degree=6)
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



