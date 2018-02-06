#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 

from time import time
from sklearn import tree
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import accuracy_score

features_train, features_test,labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

clf=tree.DecisionTreeClassifier()
t0=time()
clf.fit(features_train,labels_train)
print("training time: ",round(time()-t0,3),"s")
t0=time()
pred=clf.predict(features_test)

print("Prediction time: ",round(time()-t0,3),"s")
acc=accuracy_score(pred,labels_test)
print("accuracy: ",acc)



print("No. of POI in test set: ",len([ e for e in labels_test if e==1.0 ]))

print("No. of people in test set : ", len(labels_test))


count=0
for i in labels_test:
	if i==0.0: count=count+1
print("If the classifier predicted 0, i.e. not POI for all cases, accuracy : ",  (count*100.0)/len(labels_test)  )   

count=0
for i in range(len(labels_test)):
	if (pred[i]==1 and labels_test[i]==1):
		count=count+1
print "No. of true positives : ",count


from sklearn.metrics import *
print "Precision: ",precision_score(labels_test,pred)

print "Recall: ",recall_score(labels_test,pred)
