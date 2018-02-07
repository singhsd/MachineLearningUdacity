#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")


# My import statements
import matplotlib.pyplot as plt
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing
from time import time
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary'] # You will need to use more features
all_features = ['poi', 'email_address', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock', 'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages', 'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']




### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', np.nan)

### Task 2: Remove outliers

df[financial_features] = df[financial_features].fillna(0)
df[email_features] = df[email_features].fillna(df[email_features].median())

df = df.drop('TOTAL')
df = df[df.from_messages < 6000]
df = df[df.to_messages < 10000]

### Task 3: Create new feature(s)

df['fracion_of_messages_to_poi'] = df.from_this_person_to_poi / df.from_messages
df['fracion_of_messages_from_poi'] = df.from_poi_to_this_person / df.to_messages


### Store to my_dataset for easy export below.
my_dataset = df.to_dict('index')

features_list = [u'poi', u'salary', u'to_messages', u'deferral_payments', u'total_payments',
       u'exercised_stock_options', u'bonus', u'restricted_stock',
       u'shared_receipt_with_poi', u'expenses', u'from_messages', u'other',
       u'long_term_incentive', u'fracion_of_messages_to_poi',
		u'fracion_of_messages_from_poi']


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.cross_validation import StratifiedKFold

'''

# This module checks for the best set of parameters, is not included
# for time management issues. The classifier used is the best one

grid = {
     'base_estimator__criterion': ('gini', 'entropy'),
     'base_estimator__min_samples_leaf':range(1, 50, 5),
     'base_estimator__max_depth': range(1, 10),
     'n_estimators': range(1,10)
 }

search = GridSearchCV(AdaBoostClassifier(DecisionTreeClassifier(random_state=42), random_state=42), grid, make_scorer(f1_score), cv=StratifiedKFold(labels), n_jobs=-1)
search.fit(features, labels)

print search.best_score_
print search.best_params_

clf = search.best_estimator_




# Provided to give you a starting point. Try a variety of classifiers.
print clf

'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


best_params = {
	'n_estimators': 4, 
	'base_estimator__criterion': 'gini', 
	'base_estimator__max_depth': 3, 
	'base_estimator__min_samples_leaf': 11}

clf = AdaBoostClassifier(DecisionTreeClassifier(random_state=42), random_state=42)
clf.set_params(**best_params)


# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

clf.fit(features_train,labels_train)
pred=clf.predict(features_test)
print accuracy_score(pred,labels_test)


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


