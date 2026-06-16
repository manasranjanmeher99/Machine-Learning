import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv(r"C:\Users\ASUS\Downloads\Churn_Modelling.csv")
dataset.head()

data=dataset.iloc[:,[3,4,7,8,9,10,11,12,13]]
data

import category_encoders as ce
target_encoder=ce.TargetEncoder()
data['Geography']=target_encoder.fit_transform(data['Geography'],data['Exited'])

x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

from imblearn.over_sampling import SMOTE
s=SMOTE()
x_data,y_data=s.fit_resample(x,y)

from collections import Counter
print(Counter(y_data))

from sklearn.preprocessing import MinMaxScaler
mms=MinMaxScaler()
x_scaled=mms.fit_transform(x_data)
x_scaled


# split the dataset into training set and testing set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_data,test_size=0.25,random_state=30)


# initializing the ANN model
ann=tf.keras.models.Sequential()
ann

#Hidden Layers 
ann.add(tf.keras.layers.Dense(units=8,activation='relu')) #this is 1st hidden layer and input layer also
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
ann.add(tf.keras.layers.Dense(units=8,activation='relu'))
#Output Layer
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


# fit the train model
ann.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
ann.fit(x_train,y_train,batch_size=32,epochs=50)

ann_pred=ann.predict(x_test)
ann_pred

New_ann_pred=[]
for x in ann_pred:
    if x>=0.5:
        New_ann_pred.append(1)
    else:
        New_ann_pred.append(0)
print(New_ann_pred)

#Now we check the final accuracy of model
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
Final_score=accuracy_score(y_test,New_ann_pred)
Final_score*100

pd.DataFrame(np.c_[y_test,New_ann_pred],columns=["Actual_value","Predicted_Result"])

print(confusion_matrix(y_test,New_ann_pred))


# confusion matrix
from sklearn import metrics
import matplotlib.pyplot as plt
Confusion_Matrix=metrics.confusion_matrix(y_test,New_ann_pred)
cm_display=metrics.ConfusionMatrixDisplay(Confusion_Matrix)
cm_display.plot()
plt.show()

print(classification_report(y_test,New_ann_pred))

# visualization
data1= data[['CreditScore', 'Balance', 'EstimatedSalary',]]
sns.pairplot(data1)
plt.show()


sns.violinplot(x='NumOfProducts', y='EstimatedSalary', data=data)

features = ['CreditScore', 'NumOfProducts']
data[features].hist(figsize=(10, 4))