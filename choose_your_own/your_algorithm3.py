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
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

for n_est in range(1,101):
	for learns in range(1,100,1):
		learn=learns/10.0
		for algo in ("SAMME", "SAMME.R"):
				clf=AdaBoostClassifier(n_estimators=n_est,learning_rate=learn,algorithm=algo)
				t0=time()
				clf.fit(features_train,labels_train)
				print(n_est,learn,algo,"Training time: ",round(time()-t0,3),"s")
				t0=time()
				pred=clf.predict(features_test)
				print(n_est,learn,algo,"Prediction time: ",round(time()-t0,3),"s")
				acc=accuracy_score(pred,labels_test)
				print(n_est,learn,algo,"accuracy: ",acc)


 
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
 