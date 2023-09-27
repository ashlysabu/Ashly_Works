#load libraries
import numpy as np
import matplotlib.pyplot as mtp
import pandas as pd
data_set=pd.read_csv('C://DATASCIENCE//ashly//ashly_project//simplelinearregression.csv')
print(data_set)
x= data_set.iloc[:, :-1].values # experience, independent variable
y= data_set.iloc[:, 1].values # salary, dependent variable
x1=pd.DataFrame(x)
print("Age")
print(x1)
y1=pd.DataFrame(y)
print("Premium")
print(y1)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3,
random_state=0)
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)
y_pred= regressor.predict(x_test)
x_pred= regressor.predict(x_train)
df2 = pd.DataFrame({'Actual Y-Data': y_test, 'Predicted Y-Data':y_pred})
print(df2)
#visualizing the Training set results:
mtp.scatter(x_train, y_train, color="BLUE")
mtp.plot(x_train, x_pred, color="red")
mtp.title("Age v/s Premium")
mtp.xlabel("Age")
mtp.ylabel("Premium")
mtp.show()
