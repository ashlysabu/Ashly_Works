import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import numpy as np
data_set= pd.read_csv('C://DATASCIENCE//ashly//ashly_project//50_Startups.csv')
print(data_set.to_string())
#Extracting Independent and dependent Variable
x= data_set.iloc[:, :-1].values
y= data_set.iloc[:, 4].values
df2=pd.DataFrame(x)
print("X=")
print(df2.to_string())
df3=pd.DataFrame(y)
print("Y=")
print(df3.to_string())
#Catgorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_x= LabelEncoder()
x[:, 3]= labelencoder_x.fit_transform(x[:,3])
dt=pd.DataFrame(x)
print("--------------------")
print(dt.to_string())
print("-----------------------")
# State column
ct = ColumnTransformer([("Passenger ID", OneHotEncoder(), [3])], remainder ='passthrough')
x = ct.fit_transform(x)
""" We should not use all the dummy variables at the same time, so it
must be 1 less than the total number of
dummy variables, else it will create a dummy variable trap."""
print("----------b4 removing dummy variable-----")
dfx=pd.DataFrame(x)
print("--------------------")
print(dfx.to_string())
print("-----------------------")
#avoiding the dummy variable trap:
x = x[:, 1:]
df4=pd.DataFrame(x)
print("Updated X=")
print(df4.to_string())
# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2,random_state=0)
#Fitting the MLR model to the training set:
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)
#Predicting the Test set result;
y_pred= regressor.predict(x_test)
#To compare the actual output values for X_test with the predicted value
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.to_string())
print("Mean")
print(data_set.describe())
print("-------------------------------------")
#Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
"""You can see that the value of root mean squared error is 9137.99,
which is less than 10% of the mean value
of the expenses in all states. This means that our algorithm is very
accurate and good predictions. """
from sklearn.metrics import r2_score
# predicting the accuracy score
score=r2_score(y_test,y_pred)
print("r2 socre is ",score*100,"%")