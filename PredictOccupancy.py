
"""
Support Vector Classification to predict Occupancy at 
Electrical Vehicle Charging stations
"""
__date__   = "Jan 27, 2019"
__author__ = "Vaasudevan (https://vaasudevans.github.io)"
__info__   = "UNB GGE - BigData and DataScience Assignment-2"


# Step2 - Import the libraries and classes
from sklearn import svm, model_selection as ms
from sklearn.metrics import *
from sys import argv as arg
import pandas as pd


# Step3 - Load the Dataset
variables = ("dt", "cpId", "cpType", "status")
data = pd.read_csv(arg[1], names=variables, sep=' ')


# Step4 - Timestamp separation
data['dt'] = pd.to_datetime(data['dt'], format="%Y%m%d%H%M").astype(str)


# Step5 - Assigning number to unique values in all the columns
unique = {v: data[v].unique() for v in variables}
for v in variables:
    data[v].replace(unique[v], range(unique[v].size), inplace=True)


# Step6 - Split: 60% - Training | 40% - Testing
train, test = ms.train_test_split(data, test_size=0.4, random_state=100)


# Step7 - Make the Dataframe to be used in sklearn
Ctrain, Ctest = train.pop('status'), test.pop('status').tolist()


# Step8 - Fit the training data and create the model
clf = svm.SVC(gamma='auto')
clf.fit(train, Ctrain)


# Step9 - Test the model
predicted = clf.predict(test)


# Step10 - Accuracy report
accuracy = accuracy_score(Ctest, predicted)
report = classification_report(Ctest,
                               predicted,
                               target_names=['Occupied', 'Free'])
ConfusionMatrix = confusion_matrix(Ctest, predicted)


# Step11 - Printing the Results
print("Accuracy:", accuracy)
print("Report:\n", report)
print("ConfusionMatrix:\n", ConfusionMatrix)


# EOF
