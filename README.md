# Ad Click Prediction
Given an online ad, this application uses Apache Spark's MLlib library to implement SVM, Naive Bayes, and Logistic Regression to predict whether the ad will be clicked by a user or not.

## Data

Data was taken from [Criteo Labs](http://labs.criteo.com/) and is sample of Kaggle Display Advertising Challenge Dataset.
It can be downloaded after you accept the agreement
[http://labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/](http://labs.criteo.com/downloads/2014-kaggle-display-advertising-challenge-dataset/).

It is structured as lines of observations where first is click or no click(1,0) and rest are features.

The columns are tab separeted with the following schema:
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>

When a value is missing, the field is just empty.
There is no label field in the test set.

The training dataset consists of a portion of Criteo's traffic over a period
of 7 days. Each row corresponds to a display ad served by Criteo and the first
column is indicates whether this ad has been clicked or not.
The positive (clicked) and negatives (non-clicked) examples have both been
subsampled (but at different rates) in order to reduce the dataset size.

There are 13 features taking integer values (mostly count features) and 26
categorical features. The values of the categorical features have been hashed
onto 32 bits for anonymization purposes. 
The semantic of these features is undisclosed. Some features may have missing values.

The rows are chronologically ordered.

## Process

1) Sample is first parsed and loaded in context.
2) Transformed so it can be used in support vector machines, logistic regression, and naive bayes
3) Model created from the training data after making a 70-15-15 split for the training, validation, and testing set, respectively.
5) Iterations are performed on each three algorithms to generate the model with best parameters.
