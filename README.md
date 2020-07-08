## Why do we need Cross Validation?
When we use train test split, data for training and testing is randomly selected. Suppose the kind of data used for testing is not present in your training data set
so our model might give less accuracy. As we know we have random_state parameter to randomly select data 
for training and testing, based on that our accuracy might fluctuate

So you can't tell your stakeholders what exact accuracy your model is giving.

So to solve this problem we have cross validation.

![Image](https://media.makeameme.org/created/my-saviour-5b38e3.jpg)

# **Cross-validation** 
It is a statistical technique which involves
partitioning the data into subsets, training the data on a 
subset and use the other subset to evaluate the model’s
performance.

To reduce variability we perform multiple rounds of cross-validation with different subsets from the same data. 
We combine the validation results from these multiple rounds 
to come up with an estimate of the model’s predictive performance.

![Image](https://miro.medium.com/max/1400/1*sWQi89jsD84iYcWb4yBB3g.png)

Now let us talk about some cross validation techniques:
## Leave one out cross validation — LOOCV
Let's say we have n observations in a data set. LOOCV divides the dataset 
into n-1 observations i.e used for training and 1 observation for testing.
This process is iterated for each data point as shown below. Repeating 
this process n times generates n times Mean Square Error(MSE).

![Image](https://miro.medium.com/max/1400/1*AVVhcmOs7WCBnpNhqi-L6g.png)

The major drawback of this method is that it leads to higher variation in the testing model as we are testing against one data point. If the data point is an outlier it can lead to higher variation. Another drawback is it takes a lot of 
execution time as it iterates over ‘the number of data points’
times.

## K fold cross validation
Suppose we have 1000 observaions and set k value as 5. This K means 
5 experiments . Now this K will also decide what will be the test data
(1000/5=200). This 200 will be your test  data.
In first experiment let's say first 200 observation will be test data
In next experiment ,next 200 observation will be test data and left 
800 will be your trainining set. This goes on till 5 experiments.

![Image](https://media.geeksforgeeks.org/wp-content/uploads/crossValidation.jpg)

Each experiment will give you some accuracy,for 5 exp you
will get 5 accuracy scores . Thus you can tell your stakeholders
the average of the accuracy

**Disadvantage**

Lets say we have binary classification problem where the output is 1 or 0.
So what if in 1 of our experiments we have training set which has only 1 as output.
Such data sets are called imbalanced data set,so you might not get right kind of accuracy

![Image](https://datascience.aero/wp-content/uploads/2017/12/imbalancedata.png)

## Stratified CV
To solve the disadvantage of  K FOLD Cross Validation we have
Stratified CV.
What it does is basically it makes sure whenever training 
and testing dataset are divide atleast some no .of instances of each class are there
in both dataset. 
For example in a binary classification problem where we want to predict if a passenger on Titanic survived or not. We have two classes here Passenger either survived or did not survive. We ensure that each fold has a percentage of passengers that survived and a percentage of passengers that did not survive.

![Image](https://miro.medium.com/max/1170/1*TuWV2i98KmBxX5qkz_gX9g.png)

## Time series
In cases where you have data sets based on time stamps,randomly
splitting data won't work.Hence we need a different method .

For time series cross-validation we use forward chaining
also referred as rolling-origin.
Let us take example
D1, D2, D3 etc. are each day’s data and days highlighted in blue are used for training and days
highlighted in yellow are used for test. 

![Image](https://miro.medium.com/max/681/1*WMJCAkveTgbdBveMMMZtUg.png)

## Let's code
```ruby
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
```
Some standard imports
```ruby
iris = load_iris()
```

```ruby
l_scores = cross_val_score(LogisticRegression(), iris.data, iris.target)
np.average(l_scores)
```
Average model score when we use Logistic Regression

```ruby
d_scores = cross_val_score(DecisionTreeClassifier(), iris.data, iris.target)
np.average(d_scores)
```

Average model score when we use Decision Tree
```ruby
s_scores = cross_val_score(SVC(), iris.data, iris.target)
np.average(s_scores)
```
Average model score when we use Support Vector Machine 

Based on the score you can choose which model should be choosed for training .

That's all  from my side.
Thank You!











