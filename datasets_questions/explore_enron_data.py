#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
import os

enron_data = pickle.load(open("/home/sd/Desktop/python/ud120-projects-master/final_project/final_project_dataset.pkl", "r"))

print('No. of people in dataset: ',len(enron_data))
print('No. of features: ',len(enron_data.values()[0]))

print enron_data["SKILLING JEFFREY K"]

count=0
a=enron_data.values();
for i in range(len(enron_data)):
	if a[i]['poi']== 1 :
		count=count+1
print('No. of person of interest in E+F dataset : ',count)

text= "/home/sd/Desktop/python/ud120-projects-master/final_project/poi_names.txt"
poi_names=open(text,'r')
fr= poi_names.readlines()
print('No. of POIs: ',len(fr[2:]))
poi_names.close()


[match]=[s for s in enron_data.keys() if "PRENTICE" in s]
print("the total value of the stock belonging to James Prentice is: ",enron_data[match]['total_stock_value'])

[wesley] = [s for s in enron_data.keys() if "WESLEY" in s]
print("the number of email messages from Wesley Colwell to persons of interest is: ",enron_data[wesley]['from_this_person_to_poi'])


[skilling] = [s for s in enron_data.keys() if "SKILLING" in s]
print("the value of stock options exercised by Jeffrey Skilling is: ", enron_data[skilling]['exercised_stock_options'])

maxi=-1
ans_str=""
for i in enron_data.keys():
	if i.startswith('LAY') or i.startswith('SKILLING') or i.startswith('FASTOW'):
		print i,enron_data[i]['total_payments']
		if enron_data[i]['total_payments'] > maxi:
			maxi=enron_data[i]['total_payments']
			ans_str=i


print "Maximum total profit is earned by ",ans_str," which is ",maxi

quantifiable_salary=0
known_address=0
for i in enron_data.keys():
	if enron_data[i]['salary']!='NaN':
		quantifiable_salary+=1
	if enron_data[i]['email_address']!='NaN':
		known_address+=1;

print "No. of people with quantifiable salary : ", quantifiable_salary
print "No. of people with known email addresses : ", known_address


print "No. of people in E+F dataset having 'NaN' as their total payments is : ", sum( [1 for i in enron_data.values() if i['total_payments']=='NaN'] )
print "Percentage of people : ", 100.0*sum( [1 for i in enron_data.values() if i['total_payments']=='NaN'] ) / sum( [1 for i in enron_data.values()] )


print "No. of POIs in E+F dataset having 'NaN' fot their total payments", sum( [1 for i in enron_data.values() if (i['total_payments']=='NaN' and i['poi']== 1 ) ] )



