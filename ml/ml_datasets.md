# Machine Learning Datasets

## Training Dataset

**Training Dataset:** The sample of data used to fit the model.

The actual dataset that we use to train the model (weights and biases in the case of a Neural Network). The model sees and learns from this data.

## Validation Dataset

**Validation Dataset:** The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. The evaluation becomes more biased as skill on the validation dataset is incorporated into the model configuration.

The validation set is used to evaluate a given model, but this is for frequent evaluation. We use this data to _fine-tune_ the model hyperparameters. Therefore, the model occasionally sees this data, but never does it "Learn" from this dataset. We use the validation set results, and update higher level hyperparameters. So the validation set affects a model, but only indirectly.

## Test Dataset

**Test Dataset:** The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.

The Test dataset provides the _gold standard_ used to evaluate the model. It is only used once a model is completely trained (using the train and validation sets). The test set is generally what is used to evaluate competing models (on many Kaggle competitions, the validation set is released initially along with the training set and the actual test set is only released when the competition is about to close, and it is the result of the the model on the Test set that decides the winner). Many times the validation set is used as the test set, but it is not good practice. The test set is generally well curated. It contains carefully sampled data that spans the various classes that the model would face, when used in the real world.

## About the dataset split ratio

Now you might be looking for recommendations on how to split your dataset into Train, Validation and Test sets.

This mainly depends on two things. 1) the total number of samples in your data and 2) the actual model you are training.

Some models need substantial data to train on, so in this case you would optimize for the larger training sets. Models with very few hyperparameters will be easy to validate and tune, so you can probably reduce the size of your validation set, but if your model has many hyperparameters, you would want to have a large validation set as well (although you should also consider cross validation). Also, if you happen to have a model with no hyperparameters or ones that cannot be easily tuned, you probably do not need a validation set too!

### Note on Cross Validation

Many a times, people first split their dataset in 2 — Train and Test. Then, they set aside the Test set, and randomly choose X% of their Train dataset to be the actual Train set and the remaining (100-X)% to be the Validation set, where X is a fixed number (say 80%). The model is then iteratively trained and validated on these different sets. There are multiple ways to do this, and it is commonly known as **Cross Validation**. Basically, you use your training set to generate multiple splits of the Train and Validation sets. Cross validation avoids overfitting and is getting more and more popular, with K-fold Cross Validation being the most popular method of cross validation. Check this out for more.


## References

[About Train, Validation, and Test Sets in Machine Learning](https://towardsdatascience.com/train-validation-and-test-sets-72cb40cba9e7)
