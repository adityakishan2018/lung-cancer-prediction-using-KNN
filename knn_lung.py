import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from random import seed
from random import randrange
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# Importing dataset
X = pd.read_csv("lung_cancer_examples.csv")

#printing correlation matrix
plt.matshow(X.corr())
plt.show()

y = X.Result
   
  

# Split dataset in training and test datasets
#here we split the entire dataset in such a way that test set contains
# 40% of the entire data set
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.45, random_state=0)
print (X_train)
print (X_test)


# Instantiate the classifier
# in the entire project same Naive bayes classifier has been used for comparison
#gnb = GaussianNB()
#rfm=RandomForestClassifier(n_estimators=70, oob_score=True, n_jobs=-1,random_state=101, max_features=None, min_samples_leaf=30)
knn=KNeighborsClassifier(n_neighbors=4)

# here we are storing the important columns or attribute (in a separate array) that
# is important to determine the label class. we found it using weka application
used_features =[

    "Smokes",
   # "AreaQ",
    "Alkhol",
    "Age",
    
]

# Train classifier
#here we train our classifier using the selected important attributes stored in
# "used_features" array
# also the second parameter in gnb.fit is the column which we want to predict using #the model
# also we are storing the predicted values in "y_pred" by passing the test dataset as #input
knn.fit(
    X_train[used_features].values,
    X_train["Result"]
)
y_pred = knn.predict(X_test[used_features])

# f1 score
score = f1_score(y_pred, y_test)

#accuracy
accuracy=accuracy_score(y_test,y_pred)




# Print results
#first we display the test dataset
#then we sum all the outcomes(of test dataset) which doesn't match with predicted 
#value and display (mislabelled points)
#we find the performance % by dividing the total number outcomes which matched #with 
#predicted value by test dataset size
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
      .format(
          X_test.shape[0],
          (X_test["Result"] != y_pred).sum(),
          100*(1-(X_test["Result"] != y_pred).sum()/X_test.shape[0])
))

# print F1 SCORE
print ("The F1 score is ")
print (score)


# print Accuracy SCORE
print ("The accuracy score is ")
print (accuracy)
