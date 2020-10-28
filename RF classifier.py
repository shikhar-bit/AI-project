from matplotlib import pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

a=pd.read_csv(r'F:\mnist\mnist_test.csv')

#extracting data
b=a.iloc[5,1:].values

#reshaping data 
b=b.reshape(28,28).astype('uint8')

plt.imshow(b)

#separating labels from data 
#include all columns except the label one
pixel=a.iloc[:,1:]

label=a.iloc[:,0]

#creating 2 subsets from data
#80% train size
x_train,x_test,y_train,y_test= train_test_split(
                              pixel,label,test_size=0.4,random_state=4)

#call rf classifier
rf=RandomForestClassifier(n_estimators=100)

#fit the model
rf.fit(x_train,y_train)

#predicting test test data
pred=rf.predict(x_test)

#check prediction accuracy
s=y_test.values

#calculate correctly prediceted values
count=0
for i in range(len(pred)):
    if pred[i]==s[i]:
        count+=1

print(count)
print(count/len(pred))

