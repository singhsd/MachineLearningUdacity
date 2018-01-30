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


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

for neighbors in range(1,21):
	for criterion in ("gini","entropy"):
		for max_features in ("auto", "sqrt", "log2", None):
			for max_depth in tuple(range(1,11))+(None,):
				clf=RandomForestClassifier(n_estimators=neighbors,criterion=criterion,max_features=max_features,max_depth=max_depth)
				t0=time()
				clf.fit(features_train,labels_train)
				print(neighbors,criterion,max_features,max_depth,"Training time: ",round(time()-t0,3),"s")
				t0=time()
				pred=clf.predict(features_test)
				print(neighbors,criterion,max_features,max_depth,"Prediction time: ",round(time()-t0,3),"s")
				acc=accuracy_score(pred,labels_test)
				print(neighbors,criterion,max_features,max_depth,"accuracy: ",acc)


 
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
 