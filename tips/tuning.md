# Hyperparameter Tuning

<!-- MarkdownTOC -->

- Simple pipeline used in all the examples
- Grid Search
- Random Search
- Bayesian Search
- Visualization of parameter search \(learning rate\)
- Visualization of mean score for each iteration
- AutoML Tools for Tuning
- KerasTuner
- Optuna
- Improve Model Performance
- References

<!-- /MarkdownTOC -->


The process of searching for optimal hyperparameters is called hyperparameter tuning or _hypertuning_ which is essential in any machine learning project. 

Hypertuning helps boost performance and reduces model complexity by removing unnecessary parameters (e.g., number of units in a dense layer).

There are two types of hyperparameters:

- Model hyperparameters: influence model architecture (such as number and width of hidden layers in a DNN)

- Algorithm hyperparameters: influence the speed and quality of training (such as learning rate and activation function).


A practical guide to hyperparameter optimization using three methods: grid, random and bayesian search (with skopt)

1. Introduction to hyperparameter tuning.
2. Explanation about hyperparameter search methods.
3. Code examples for each method.
4. Comparison and conclusions.

### Simple pipeline used in all the examples

```py
    from sklearn.pipeline import Pipeline #sklearn==0.23.2
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import make_column_transformer
    from lightgbm import LGBMClassifier
    
    tuples = list()
    
    tuples.append((Pipeline([
            ('scaler', StandardScaler()),
        ]), numeric_var))
    
    tuples.append((Pipeline([
            ('onehot', OneHotEncoder()),
        ]), categorical_var))
    
    preprocess = make_column_transformer(*tuples)
    
    pipe = Pipeline([
        ('preprocess', preprocess),
        ('classifier', LGBMClassifier())
        ])
```


### Grid Search

The basic method to perform hyperparameter tuning is to try all the possible combinations of parameters.

### Random Search

in randomized search, only part of the parameter values are evaluated. 

The parameter values are sampled from a given list or specified distribution. 

The number of parameter settings that are sampled is given by `n_iter`. 

Sampling without replacement is performed when the parameters are presented as a list (similar to grid search), but if the parameter is given as a distribution then sampling with replacement is used (recommended).

The advantage of randomized search is that you can extend your search limits without increasing the number of iterations. You can also use random search to find narrow limits to continue a thorough search in a smaller area.

### Bayesian Search

The main difference with Bayesian search is that the algorithm optimizes its parameter selection in each round according to the previous round score. Thus, the algorithm optimizes the choice and theoretically reaches the best parameter set faster than the other methods which means that this method will choose only the relevant search space and discard the range of values that will most likely not deliver the best solution. 

Thus, Bayesian search can be beneficial when you have a large amount of data and/or the learning process is slow and you want to minimize the tuning time.

```py
    from skopt import BayesSearchCV
    
    # Bayesian
    n_iter = 70
    
    param_grid = {
        "classifier__learning_rate": (0.0001, 0.1, "log-uniform"),
        "classifier__n_estimators": (100,  1000) ,
        "classifier__max_depth": (4, 400) 
    }
    
    reg_bay = BayesSearchCV(estimator=pipe,
                        search_spaces=param_grid,
                        n_iter=n_iter,
                        cv=5,
                        n_jobs=8,
                        scoring='roc_auc',
                        random_state=123)
    
    model_bay = reg_bay.fit(X, y)
```


### Visualization of parameter search (learning rate)

### Visualization of mean score for each iteration



## AutoML Tools for Tuning

### [KerasTuner](https://keras.io/keras_tuner/)

KerasTuner is an easy-to-use, scalable hyperparameter optimization framework that solves the pain points of hyperparameter search. 

The process of selecting the right set of hyperparameters for your machine learning (ML) application is called _hyperparameter tuning_ or _hypertuning_.

Hyperparameters are the variables that govern the training process and the topology of an ML model. These variables remain constant over the training process and directly impact the performance of your ML program. Hyperparameters are of two types:

- **Model hyperparameters** which influence model selection such as the number and width of hidden layers

- **Algorithm hyperparameters** which influence the speed and quality of the learning algorithm such as the learning rate for Stochastic Gradient Descent (SGD) and the number of nearest neighbors for a k Nearest Neighbors (KNN) classifier



### Optuna

Optuna is an automatic hyperparameter optimization software framework designed for machine learning. 

Optuna is framework agnostic, so you can use it with any machine learning or deep learning framework



## Improve Model Performance

[Profiling Neural Networks to improve model training and inference speed](https://pub.towardsai.net/profiling-neural-networks-to-improve-model-training-and-inference-speed-22be473492bf)

[How to Speed Up XGBoost Model Training](https://towardsdatascience.com/how-to-speed-up-xgboost-model-training-fcf4dc5dbe5f?source=rss----7f60cf5620c9---4)



## References

[How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)

[A Practical Introduction to Grid Search, Random Search, and Bayes Search](https://towardsdatascience.com/a-practical-introduction-to-grid-search-random-search-and-bayes-search-d5580b1d941d)

[Hyperparameter Tuning Methods](https://towardsdatascience.com/bayesian-optimization-for-hyperparameter-tuning-how-and-why-655b0ee0b399)

[Hyperparameter Tuning with KerasTuner and TensorFlow](https://towardsdatascience.com/hyperparameter-tuning-with-kerastuner-and-tensorflow-c4a4d690b31a)

[Introduction to the Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner)



[How to Grid Search Deep Learning Models for Time Series Forecasting](https://machinelearningmastery.com/how-to-grid-search-deep-learning-models-for-time-series-forecasting/)

[Scikit-Optimize for Hyperparameter Tuning in Machine Learning](https://machinelearningmastery.com/scikit-optimize-for-hyperparameter-tuning-in-machine-learning/)

