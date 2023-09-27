import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
# importing datasets
data_set = pd.read_csv('C://DATASCIENCE//ashly//ashly_project//house.csv')
print(data_set)
#Extracting Independent and dependent Variable
x= data_set.iloc[:, 1:2].values
y= data_set.iloc[:, 2].values
#convert to dataframe
df1=pd.DataFrame(x)
df2=pd.DataFrame(y)
print("bedroom_count")
print(df1.to_string())
print("price")
print(df2.to_string())

#Fitting the Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regs= LinearRegression()
lin_regs.fit(x,y)
from sklearn.preprocessing import PolynomialFeatures
#The parameter value(degree= 2) depends on our choice. We can choose it according to our Polynomial features.
poly_regs= PolynomialFeatures(degree= 2)
#we are converting our feature matrix into polynomial feature matrix, andthen fitting it to the Polynomial regression
x_poly= poly_regs.fit_transform(x)
lin_reg_2 =LinearRegression()
lin_reg_2.fit(x_poly, y)
#print("done")
#Visualizing the result for Linear regression:
#Visulaizing the result for Linear Regression model
mtp.scatter(x,y,color="red")
mtp.plot(x,lin_regs.predict(x), color="blue")
mtp.title("Bluff detection model(Linear Regression)")
mtp.xlabel("age")
mtp.ylabel("price")
mtp.show()
#Visualizing the result for Polynomial Regression
"""Here we will visualize the result of Polynomial regression model,
code
for which is little different from the above model."""
#Visulaizing the result for Polynomial Regression
mtp.scatter(x,y,color="blue")
mtp.plot(x, lin_reg_2.predict(poly_regs.fit_transform(x)), color="red")
mtp.title("Bluff detection model(Polynomial Regression)")
mtp.xlabel("age")
mtp.ylabel("price")
mtp.show()
#Predicting the final result with the Linear Regression model:
"""Now, we will predict the final output using the Linear regression
model to see whether an employee is saying truth or bluff. So, for this,
we will use the predict() method and will pass the value 6.5. Below is
the code for it:"""
lin_pred = lin_regs.predict([[6.5]])
print(lin_pred)
#Predicting the final result with the Polynomial Regression model:
"""Now, we will predict the final output using the Polynomial
Regression model to compare
with Linear model. Below is the code for it:"""
poly_pred = lin_reg_2.predict(poly_regs.fit_transform([[6.5]]))
print(poly_pred)