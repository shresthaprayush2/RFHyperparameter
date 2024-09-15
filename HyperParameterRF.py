#Hyperparameters are the tunable parameters that help tune and customize the Scikit learn model
#Different paramter have different hyperparameters
#In this tutorial we will explore the hyperparameters of SVC algorithm
#SVC algorithm is the classification alogirthm

#Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

trainX,testX,trainY,testY = train_test_split(X,y,random_state=12)
allAccuracy = []

#Printing original data
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
plt.savefig('ogdata.png')


#Creating a function to plot the data
def plotRandomForest(title,svc):
  # create a mesh to plot in
  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  h = (x_max / x_min)/100
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))
  plt.subplot(1, 1, 1)
  Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
  plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired)
  plt.xlabel('Sepal length')
  plt.ylabel('Sepal width')
  plt.xlim(xx.min(), xx.max())
  plt.title(title)
  plt.savefig(f'{title}.png')



nestimators = [10,35,50,80]
for estimator in nestimators:
  rf = RandomForestClassifier(n_estimators=estimator).fit(trainX, trainY)
  predictedY = rf.predict(testX)
  accuracy = accuracy_score(predictedY,testY)
  allAccuracy.append(accuracy*100)

  plotRandomForest('Estimator=' + str(estimator),rf)
accuracy_details = pd.Series(allAccuracy,index=['Trees : 10','Trees : 35','Trees : 50','Trees : 80'])
print(accuracy_details)

#Clearing the same array becuse i am lazy
allAccuracy.clear()
maxDepths = [1,5,10,24,100]
for ms in maxDepths:
   rf = RandomForestClassifier(max_depth=ms).fit(trainX,trainY)
   predictedY = rf.predict(testX)
   accuracy = accuracy_score(predictedY, testY)
   allAccuracy.append(accuracy * 100)
   plotRandomForest('maxDepth=' + str(ms),rf)

print("Max Depth")
accuracy_details = pd.Series(allAccuracy, index=['Max Depth : 1', 'Max Depth : 5', 'Max Depth : 10', 'Max Depth : 24','Max Depth : 100'])
print(accuracy_details)

allAccuracy.clear()
min_samples_leafs=[2,10,20,30,100]
for ml in min_samples_leafs:
   rf = RandomForestClassifier(min_samples_leaf=ml).fit(trainX, trainY)
   predictedY = rf.predict(testX)
   accuracy = accuracy_score(predictedY, testY)
   allAccuracy.append(accuracy * 100)
   plotRandomForest('SampleLeaf=' + str(ml),rf)

print("Minimum Sample Leaf")
accuracy_details = pd.Series(allAccuracy, index=['Minimum Sample Leaf : 2', 'Minimum Sample Leaf : 10', 'Minimum Sample Leaf : 20', 'Minimum Sample Leaf : 30','Minimum Sample Leaf : 100'])
print(accuracy_details)

allAccuracy.clear()
max_leaf_nodes=[2,10,20,30,100]
for mln in max_leaf_nodes:
   rf = RandomForestClassifier(max_leaf_nodes=mln).fit(trainX, trainY)
   predictedY = rf.predict(testX)
   accuracy = accuracy_score(predictedY, testY)
   allAccuracy.append(accuracy * 100)
   plotRandomForest('MaxLeafNodes=' + str(mln),rf)

print("Max Leaf Nodes")
accuracy_details = pd.Series(allAccuracy, index=['Max Leaf Nodes : 2', 'Max Leaf Nodes : 10', 'Max Leaf Nodes : 20', 'Max Leaf Nodes : 30','Max Leaf Nodes : 100'])
print(accuracy_details)

allAccuracy.clear()
min_sample_split=[2,10,20,30,100]
for mss in min_sample_split:
   rf = RandomForestClassifier(min_samples_split=mss).fit(trainX, trainY)
   predictedY = rf.predict(testX)
   accuracy = accuracy_score(predictedY, testY)
   allAccuracy.append(accuracy * 100)
   plotRandomForest('MinimumSampleSplit' + str(mss),rf)

print("Minimum Sample Split")
accuracy_details = pd.Series(allAccuracy, index=['Minimum Sample Split : 2', 'Minimum Sample Split : 10', 'Minimum Sample Split : 20', 'Minimum Sample Split : 30','Minimum Sample Split : 100'])
print(accuracy_details)

from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier()
gridSearch = GridSearchCV(estimator=model,param_grid={'n_estimators':nestimators,'max_depth':maxDepths,'min_samples_leaf':min_samples_leafs,'max_leaf_nodes':max_leaf_nodes,'min_samples_split':min_sample_split},cv=4)
gridSearch.fit(X,y)

print(f'The best parameter is {gridSearch.best_params_}')
print(f'The best score is {gridSearch.best_score_}')
print(f'The best estimator is {gridSearch.best_estimator_}')



