#importing dependencies
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_boston
from matplotlib import pyplot as plt
#check accuracy
from sklearn.metrics import r2_score


#understanding data
boston=datasets.load_boston()

#access data
for index,name in enumerate(boston.feature_names):
    print(index,name)


#reshaping data
b=boston.data

x=b[:,5].reshape(-1,1)
y=boston.target.reshape(-1,1)



#regression
from sklearn.linear_model import LinearRegression

#creating a regression model
reg=LinearRegression()

#fit the model
reg.fit(x,y)

#predicton
pred=reg.predict(x)

#plot predicted line
plt.plot(x,pred,color='blue')
print('linear acc'+str(r2_score(pred,y)))

#using polynomial regression
from sklearn.preprocessing import PolynomialFeatures
#to allow merging...powers of polynomials
from sklearn.pipeline import make_pipeline

model=make_pipeline(PolynomialFeatures(3), reg)
model.fit(x,y)

pred=model.predict(x)

#check matplotlib
plt.scatter(x,y,color='green')
plt.plot(x,pred,color='red')
plt.xlabel('no. of rooms')
plt.ylabel('cost of house')
plt.show()


#pred
print('polynomial acc'+str(r2_score(pred,y)))




