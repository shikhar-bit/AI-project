import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from matplotlib import pyplot as plt
import seaborn as sns

cancer=load_breast_cancer()

cancer_df = pd.DataFrame(np.c_[cancer['data'],cancer['target']],
            columns = np.append(cancer['feature_names'], ['target']))

sns.countplot(cancer_df['target'])
#correlation of various parameters with target
df2=cancer_df.drop(['target'],axis=1)
plt.figure(figsize=(16,5))
sns.barplot(df2.corrwith(cancer_df.target).index,df2.corrwith(cancer_df.target))

#separating target from test and 
#segregating as  X and Y 
x=cancer_df.drop(['target'],axis=1)
y=cancer_df['target']

#split
from sklearn.model_selection import train_test_split
X_train,X_validation,Y_train,Y_validation=train_test_split(x,y,test_size=0.2)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_fit=sc.fit_transform(X_train)
x_test_fit= sc.transform(X_validation)

#Support Vector Classifier
from sklearn.svm import SVC
svc_classifier=SVC()

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
kfold=StratifiedKFold(n_splits=10,shuffle=True ,random_state=1)
cv_result=cross_val_score(SVC(gamma='auto'), X_train,Y_train,cv=kfold, scoring='accuracy')

#u can also fit and predict raw unscaled data....get a 
#low accuracy score
svc_classifier.fit(x_train_fit, Y_train)
y_pred=svc_classifier.predict(x_test_fit)

from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import accuracy_score

print('accuracy score is %s' %(accuracy_score(Y_validation,y_pred)))

