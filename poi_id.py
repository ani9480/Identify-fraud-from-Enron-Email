#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt



### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock',
                      'director_fees']

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages',
                  'from_this_person_to_poi', 'shared_receipt_with_poi'] 

POI_label = ['poi'] 


total_features_list = POI_label + financial_features + email_features


features_list = total_features_list
print(total_features_list)    #Check if all the features has been added
print("\t")

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

import random
name, assets = random.choice(list(data_dict.items()))

print ("There are {0} executives in Enron Dataset.".format(len(data_dict.keys())) )
print("\t")

### Task 2: Remove outliers

## So here I first plot salary vs bonus to check if their are any outliers and if so i would remove them 
## in my next steps

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)
    

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


### Removing outliers
data_dict.pop('TOTAL',0)
data_dict.pop('The Travel Agency In the Park',0)

data_dict_list = []

for key in data_dict:
    val = data_dict[key]['salary']
    if val == 'NaN':
        continue
    data_dict_list.append((key, int(val)))



# Sorting the data_dict list and storing the top 5 executives with highest salary and bonus
data_dict_list_final = (sorted(data_dict_list,key=lambda x:x[1],reverse=True)[:5])
### print top 5 salaries
print data_dict_list_final
print("\t")

    
# Again checking the distribution of the two selected features 
features = ["salary", "bonus"]
data = featureFormat(data_dict, features, remove_any_zeroes=True)

for point in data:
    salary = point[0]
    bonus = point[1]
    plt.scatter( salary, bonus )

plt.xlabel("salary")
plt.ylabel("bonus")
plt.show()


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

def add_features(data_dict):
    """
    Given the data dictionary of people with features, adds some features to
    """
    for name in data_dict:

        # Add ratio of POI messages to total.
        try:
            total_messages = data_dict[name]['from_messages'] + data_dict[name]['to_messages']
            poi_related_messages = data_dict[name]["from_poi_to_this_person"] +\
                                    data_dict[name]["from_this_person_to_poi"] +\
                                    data_dict[name]["shared_receipt_with_poi"]
            poi_ratio = 1.* poi_related_messages / total_messages
            data_dict[name]['poi_ratio_messages'] = poi_ratio
            data_dict[name]['poi_ratio_messages_squared'] = poi_ratio ** 2
        except:
            data_dict[name]['poi_ratio_messages'] = 'NaN'
        
        # Add ratio of restricted stock value to total stock value.
        try:
            
            restricted_stock_ratio = float(data_dict[name]['restricted_stock']) / float(data_dict[name]['total_stock_value'])
            data_dict[name]['restricted_stock_ratio'] = round(restricted_stock_ratio, 2)
        except:
            data_dict[name]['restricted_stock_ratio'] = 'NaN'
            
        # Add ratio of money spent.    
        try:
            money_ratio = float(data_dict[name]['expenses'])/float(data_dict[name]['salary'])
            data_dict[name]['money_ratio'] = round(money_ratio, 2)
        except:
            data_dict[name]['money_ratio'] = 'NaN'

    # print "finished"

    return data_dict

## Checking Nan values

import operator
from pprint import pprint


## Check the % of NaNs for each feature - for general understanding.
def check_nan_features(dict):
    """
    Checks the percentage of NaNs for each feature in the data.
    Args:
        dict: data dictionary
    Returns: dict of all features with the percentage of NaNs
    """
    final = {}
    ftrs_list = dict[dict.keys()[0]].keys()
    cnt = float(len(data_dict))
    for i in ftrs_list:
        n = 0.0
        for k, v in dict.items():
            if v[i] == 'NaN':
                n += 1.0
        final[i] = (n / cnt) * 100
    return final

nans_check = check_nan_features(data_dict)
sorted_nans_check = sorted(nans_check.items(), key=operator.itemgetter(1), reverse = True)
print '\nNaN Checks for Each Feature'
pprint (sorted_nans_check)

my_dataset = add_features(data_dict)

unwanted_features = ['restricted_stock_deferred', 'director_fees', 'loan_advances']

features_list = [feature for feature in total_features_list if feature not in unwanted_features]

print features_list
print("\t")


### Extract features and labels from dataset for local testing

print "All features"
print total_features_list
data = featureFormat(my_dataset, total_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


print("\t")

from sklearn.feature_selection import SelectKBest, f_classif
kbest = SelectKBest(k=10)
selected_features = kbest.fit_transform(features,labels)
features_selected=[total_features_list[i+1] for i in kbest.get_support(indices=True)]
print 'Features selected by SelectKBest:'
print features_selected
print "\t"

indices = kbest.get_support(True)
#print kbest.scores_
print "Features and their Scores"

for index in indices:    
    print 'features: %s score: %f' % (total_features_list[index+1], kbest.scores_[index])
print "\t"

## At first i thought of selecting features via KBest but later i used pipeline to get the
## desired result so no need of this step
    
    





### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)

from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)




# Provided to give you a starting point. Try a variety of classifiers.
###NaiveBayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import accuracy_score

skb = SelectKBest(k = 10)
g_clf =  Pipeline(steps=[('scaling',scaler),("SKB", skb), ("NaiveBayes", GaussianNB())])
g_clf.fit(features_train, labels_train)
pred = g_clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
accuracy = g_clf.score(features_test, labels_test)
print "accuracy GaussianNB",accuracy
pre = precision_score(labels_test, pred)
print "pre",pre
rec = recall_score(labels_test, pred)
print "rec",rec
print("\t")



###Decision Tree
from sklearn import tree
d_clf = tree.DecisionTreeClassifier()
parameters = {'criterion': ['gini', 'entropy'],
'min_samples_split': [5],
'max_depth': [None, 2, 5, 10],
'min_samples_leaf': [1, 5, 10],
'max_leaf_nodes': [None, 5, 10, 20]}
d_clf = GridSearchCV(d_clf, parameters)
d_clf.fit(features_train, labels_train)
d_pred= d_clf.predict(features_test)
accuracy_d = d_clf.score(features_test, labels_test)
print 'DecisionTree accuracy ' + str(accuracy_d)
print "DT Recall Score " + str(recall_score(labels_test, d_pred))
print "DT Precision Score " + str(precision_score(labels_test, d_pred))
print("\t")



### Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier(random_state = 48)
params_random_tree = {  "n_estimators":[8],
                      "criterion" : ['gini']
                    }
rf_clf = GridSearchCV(rf_clf, params_random_tree)
rf_clf.fit(features_train, labels_train)
rf_pred= rf_clf.predict(features_test)
accuracy_rf = rf_clf.score(features_test, labels_test)
print 'Random Forest accuracy ' + str(accuracy_rf)
print "RF Recall Score " + str(recall_score(labels_test, rf_pred))
print "RF Precision Score " + str(precision_score(labels_test, rf_pred))
print("\t")



###ADABOOST
from sklearn.ensemble import AdaBoostClassifier
ab_clf = AdaBoostClassifier()
parameters = {'n_estimators': [10, 20, 30, 40, 50],
'algorithm': ['SAMME', 'SAMME.R'],
'learning_rate': [.5,.8, 1, 1.2, 1.5]}
ab_clf = GridSearchCV(ab_clf, parameters)
ab_clf.fit(features_train, labels_train)
ab_pred= ab_clf.predict(features_test)
accuracy_ab = ab_clf.score(features_test, labels_test)
print 'ADABOOST accuracy ' + str(accuracy_ab)
print "AB Recall Score " + str(recall_score(labels_test, ab_pred))
print "AB Precision Score " + str(precision_score(labels_test, ab_pred))



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!


from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

SKB = SelectKBest()
pipe = Pipeline(steps=[("SKB", SKB), ('PCA', PCA(svd_solver='randomized')), ("NaiveBayes", GaussianNB())])
print sorted(pipe.get_params().keys())

# define the parameter grid for SelectKBest,
# using the name from the pipeline followed by 2 underscores:
parameters = {'SKB__k': range(4,7),
              'SKB__score_func': [f_classif],
              'PCA__n_components': [2],
              'PCA__whiten': [True]}


sss = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
# use 'StratifiedShuffleSplit' as the cross-validation method: i.e. 'cv=sss':
gs = GridSearchCV(pipe, param_grid = parameters, cv=sss, scoring = 'f1')

gs.fit(features, labels)
clf = gs.best_estimator_

print 'best algorithm'


from tester import test_classifier
test_classifier(clf, my_dataset, features_list)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)