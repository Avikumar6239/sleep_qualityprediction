# Implementation of Sleep Quality Prediction through Logistic Regression

#import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math

sleep_data= pd.read_csv('December Sleep data - Sheet1.csv')
print(sleep_data.head(10))

#Analizing the Data
sns.countplot(x="SLEEP SCORE",data= sleep_data)
plt.show()
sns.countplot(x="SLEEP SCORE",hue="DEEP SLEEP",data= sleep_data)
plt.show()
sns.countplot(x="SLEEP SCORE",hue="REM SLEEP",data= sleep_data)
plt.show()
sns.countplot(x="SLEEP SCORE",hue="Minutes of Sleep",data= sleep_data)
plt.show()
sleep_data["DEEP SLEEP"].plot.hist()
sleep_data.info()

#Data Wrangling
sleep_data.isnull()
sleep_data.dropna()
sleep_data.drop(['DECEMBER','DATE','SLEEP TIME','HOURS OF Sleep','Unnamed: 8','HEART RATE BELOW RESTING'],axis=1,inplace=True)
print(sleep_data.head(10))
sns.boxplot(x="REM SLEEP",y="DEEP SLEEP",data=sleep_data)

#Train and Testing
X= sleep_data.drop("SLEEP SCORE",axis=1)
Y= sleep_data["SLEEP SCORE"]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=0)

from sklearn.linear_model import LogisticRegression
logmodel= LogisticRegression()
logmodel.fit(X_train,Y_train)
prediction= logmodel.predict(X_test)


from sklearn.metrics import classification_report
clas=classification_report(Y_test,prediction)
print(clas)

from sklearn.metrics import confusion_matrix
cnf=confusion_matrix(Y_test,prediction)
print("Confusion matrix:-")
print(cnf)

print("--------------------------------------------------------")
print("Testing Data:-")
print(X_test)
print("Prediction: ",prediction)
print("--------------------------------------------------------")

# Accuracy Check
from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test,prediction)
print("Accuracy: ",acc)