# Support Vector Classification
Python script to predict Occupancy using SVM in [sklearn](https://scikit-learn.org/stable/modules/svm.html) library.

**Note:**
This script uses _60% Training_ and _40% testing_ of the data file.
Random seed used in _train_test_split_ is 100.
Tested in *python: 3.5, Pandas: 0.23.4, scikit-learn: 0.20.2*
License: GNU-GPL

**Installing Dependencies**
```
pip3 install pandas scikit-learn
```
**Data-Insight**
```
$ head Data/430pm_Managed.txt
201611071630 CP:C96GB CHAdeMO Occ
201611071630 CP:C96GB FastAC43 Free
201611071630 CP:C96GB ComboCCS Occ
201611071630 CP:C9QKW CHAdeMO Occ
201611071630 CP:C9QKW FastAC43 Occ
201611071630 CP:C9QKW ComboCCS Occ
201611071630 CP:RC13 CHAdeMO Free
201611071630 CP:RC13 FastAC43 Free
201611071630 CP:RC13 ComboCCS Occ
201611071630 CP:C85WS CHAdeMO Free
```
**Usage**
```
>>> python3 PredictOccupancy.py Data/430pm_Managed.txt

Accuracy: 0.714516129032258
Report:
               precision    recall  f1-score   support

    Occupied       0.51      0.19      0.28       178
        Free       0.74      0.93      0.82       442

   micro avg       0.71      0.71      0.71       620
   macro avg       0.62      0.56      0.55       620
weighted avg       0.67      0.71      0.67       620

ConfusionMatrix:
 [[ 34 144]
 [ 33 409]]
```

