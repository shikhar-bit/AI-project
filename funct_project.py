
    
import pandas
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.ensemble import VotingClassifier

x='F:\IRIS.csv'
a=read_csv(x)

#univariate plots -box and whisker plots
a.plot(kind='box', subplots=True, layout=(2,2), sharex=False,sharey=False)
pyplot.show()

#creating a validation dataset
#splitting dataset
array=a.values
X=array[:,0:4]
Y=array[:,4]
X_train,X_validation,Y_train,Y_validation=train_test_split(X,Y,test_size=0.2)

#creating models
models=[]
#Logistic Regression
models.append(('LR',LogisticRegression(solver='liblinear',multi_class='ovr')))

#Linear Discriminant Analysis
models.append(('LDA',LinearDiscriminantAnalysis()))

#K-nearest neaighbors
models.append(('KNN' ,KNeighborsClassifier()))

#Classification and Regression Trees
models.append(('NB', GaussianNB()))

#Support Vector Machines
models.append(('SVM' , SVC(gamma='auto')))

#evaluate created models
results=[]
names=[]
for name , model in models:
    kfold=StratifiedKFold(n_splits=10,shuffle=True ,random_state=1)
    cv_results= cross_val_score(model, X_train,Y_train,cv=kfold,scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s : %f (%f)' % (name, cv_results.mean(), cv_results.std()))
    
#compare models

pyplot.boxplot(results, labels=names)
pyplot.title('Algorithm Comparison')
pyplot.show()

#make predictions on LDA
model=LinearDiscriminantAnalysis()
model.fit(X_train,Y_train)
pred=model.predict(X_validation)

#evaluate predictions
print(accuracy_score(Y_validation, pred))
print(confusion_matrix(Y_validation,pred))
print(classification_report(Y_validation, pred))
