import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset= pd.read_csv(r"D:\Data Science Daily Notes\03.06.26\Churn_Modelling.csv")

x= dataset.iloc[:,3:-1].values
y= dataset.iloc[:,-1].values

print(x)
print(y)

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
x[:,2]= le.fit_transform(x[:,2])

print(x)

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct= ColumnTransformer(transformers=[('encoder', 
                                     OneHotEncoder(), [1])],
                                      remainder='passthrough')

x= np.array(ct.fit_transform(x))

#splitting the dataset
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.20, random_state=0)


from xgboost import XGBClassifier
classifier= XGBClassifier()
classifier.fit(x_train, y_train)

y_pred= classifier.predict(x_test)

#confusion mtrix
from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac= accuracy_score(y_test, y_pred)
print("Accuracy score-",ac)

from sklearn.metrics import classification_report
cr= classification_report(y_test, y_pred)
print(cr)

bias= classifier.score(x_train,y_train)
print("bias-",bias)

variance= classifier.score(x_test,y_test)
print("Variance-",variance)