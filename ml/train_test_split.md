## Train-Test Split

A key step in ML is the choice of model.  

> Split first, normalize later.

A train-test split conists of the following:

1. Split the dataset into training, validation and test set

2. We normalize the training set only (fit_transform). 

3. We normalize the validation and test sets using the normalization factors from train set (transform).

Suppose we fit the model with the train set while evaluating with the test set, we would obtain only a _single_ sample point of evaluation with one test set. 

If we have two models and found that one model is better than another based on the evaluation, how can we know this is not by chance?

**Solution:** the train-validation-test split


## Train-Validation-Test Split

Here are the steps for a train-validation-test split:

1. The model is fit on the train data set. 

2. The fitted model is used to predict the responses for the observations on the validation set. 

3. The test set is used to provide an unbiased evaluation of the final model that has been fit on the train dataset. 

If the data in the test set has never been used in training (such as cross-validation), the test set is also called a _holdout_ data set.


The reason for such practice is the concept of preventing _data leakage_ which is discussed below. 


What we should care about is the evaluation metric on the _unseen data_. 

Therefore, we need to keep a slice of data from the entire model selection and training process and save it for the final evaluation called the test set. 

The process of _cross-validation_ is the following:

1. The train set is used to train a few candidate models. 

2. The validation set is used to evaluate the candidate models. 

3. One of the candidates is chosen. 

4. The chosen model is trained with a new train set.

5. The final trained model is evaluated using the test set. 

The dataset for evaluation in step 5 and the one we used in steps 2 are different because we do not want _data leakage_. 

If the test and validation sets were the same, we would see the same score that we have already seen from cross validation or the test score would be good because it was part of the data we used to train the model and we  adapted the model for that test dataset.

Thus, we make use of the test dataset that was never used in previous steps (holdout set) to evaluate the performance on unseen data which is called _generalization_.  

