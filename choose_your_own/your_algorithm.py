#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################
 

### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


# KNeighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


for neighbors in range(1,11):
	for weights in ('uniform','distance'):
		for algorithm in ("auto","ball_tree","kd_tree","brute"):
			clf=KNeighborsClassifier(neighbors,weights,algorithm)
			t0=time()
			clf.fit(features_train,labels_train)
			print(neighbors,weights,algorithm,"Training time: ",round(time()-t0,3),"s")
			t0=time()
			pred=clf.predict(features_test)
			print(neighbors,weights,algorithm,"Prediction time: ",round(time()-t0,3),"s")
			acc=accuracy_score(pred,labels_test)
			print(neighbors,weights,algorithm,"accuracy: ",acc)


 
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
 