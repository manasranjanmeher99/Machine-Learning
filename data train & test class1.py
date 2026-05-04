import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset= pd.read_csv(r"C:\Users\ASUS\Downloads\Data.csv")

# print(dataset)
x= dataset.iloc[:, :-1].values

#Dependent variable
y= dataset.iloc[:,3].values

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()

imputer = imputer.fit(x[:,1:3])
x[:,1:3]= imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder
lebelencoder_x = LabelEncoder()

x[:,0]=lebelencoder_x.fit_transform(x[:,0])

lebelencoder_y = LabelEncoder()
y= lebelencoder_y.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8, test_size=0.2, random_state=0)


