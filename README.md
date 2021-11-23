# Module_12_Challenge
### FTBC Module 12: Supervised Learning

$~$

# Analysis Report

## Overview of the Analysis

This analysis was completed through building a supervised learning model that can identify the creditworthiness of new loan borrowers. The data consists of historical lending activity from a peer-to-peer lending services company. This data is inherently imbalanced, as the number of healthy loans made by a lender will outweigh the number of risky loans. Otherwise, the lender would not be in business for long.

The variable of interest for predictions, `loan_status` (`y`), was first tallied using the `value_counts` function. 0 corresponds with healthy loans, 1 with risky ones. After the data was fitted to a logistic regression model, the `X` features and `y` were each split using `train_test_split`. An accuracy score, confusion matrix, and classification report were generated. With consideration for the imbalanced dataset, a second analysis was completed using the `RandomOversampler`. The model was reevaluated and the two sets of performance reports were used to compare the model's performance with each dataset.

## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1: Original (imbalanced) Data

`value_counts()`:
```
0    75036
1     2500
```

Accuracy:
```
0.9520479254722232
```

Confusion matrix:
```
array([[18663,   102],
       [   56,   563]], dtype=int64)
```

Classification report:
```
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.91      1.00      0.95      0.91     18765
          1       0.85      0.91      0.99      0.88      0.95      0.90       619

avg / total       0.99      0.99      0.91      0.99      0.95      0.91     19384
```

* Machine Learning Model 2: Resampled Data

`value_counts()`:
```
0    56271
1    56271
```

Accuracy:
```
0.9936781215845847
```

Confusion matrix:
```
array([[18649,   116],
       [    4,   615]], dtype=int64)
```

Classification report:
```
                   pre       rec       spe        f1       geo       iba       sup

          0       1.00      0.99      0.99      1.00      0.99      0.99     18765
          1       0.84      0.99      0.99      0.91      0.99      0.99       619

avg / total       0.99      0.99      0.99      0.99      0.99      0.99     19384
```

## Summary

Most performance indicators improved in the resampled data model as compared to the original one. Accuracy is higher (95.20% vs. 99.37%), the confusion matrix's false positives are better (0.30% vs. 0.02%), but the false negatives are still roughly equal (18.12% vs. 18.86%). This is reflected in the precision metric within the classification report (0.85 vs. 0.84), which leads to the conclusion that neither model sufficiently predicted the 'y' target column's test values.