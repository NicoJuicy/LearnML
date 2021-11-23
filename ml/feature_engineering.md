# Feature Engineering

<!-- MarkdownTOC -->

- Overview
- Dimensionality Reduction
- Feature Importance
    - Dataset loading and preparation
    - Method 1: Obtain importances from correlation coefficients
    - Method 2: Obtain importances from a tree-based model
    - Method 3: Obtain importances from PCA loading scores
- Feature Engineering Techniques
- Transform Target Variables for Regression
    - Importance of Data Scaling
    - How to Scale Target Variables?
    - Automatic Transform of the Target Variable
    - Complete Regression Example
- References
    - Time Series Data Preparation

<!-- /MarkdownTOC -->

## Overview

[Data Preparation](./data_prep.md)

Feature engineering techniques for machine learning are a fundamental topic in machine learning but one that is often overlooked or deceptively simple.

Feature engineering consists of various processes:

- **Exploratory Data Analysis:** Exploratory data analysis (EDA) is a powerful and simple tool that can be used to improve your understanding of your data by exploring its properties. 

  EDA is often applied when the goal is to create hypotheses or find patterns in the data. 

  EDA is often used on large amounts of qualitative or quantitative data that have not been analyzed before.

- **Transformations:** Feature transformation is simply a function that transforms features from one representation to another. 

  The goal is to plot and visualize the data. If something is not adding up with the new features we can reduce the number of features used, speed up training, or increase the accuracy of a model.

- **Feature Selection:** The process of creating new variables that will be most helpful for our model which can include adding or removing some features. 

- **Feature Extraction:** The process of extracting features from a data set to identify useful information. 

  Without distorting the original relationships or other information, we compress the amount of data into manageable quantities for algorithms to process.

- **Benchmark:** A Benchmark Model is the most user-friendly, dependable, transparent, and interpretable model against which you can measure your final model. 

  It is a good idea to run test datasets to see if your new machine learning model outperforms a recognized benchmark which are often used as measures for comparing the performance of different ML models. 


## Dimensionality Reduction

Dimensionality reduction refers to the process of reducing the number of attributes in a dataset while keeping as much of the variation in the original dataset as possible. 

Dimensionality reduction is a data preprocessing step, so it is done before training the model.

There are two main methods for reducing dimensionality:

- In **feature selection**, we only keep the most important features in the dataset and remove the redundant features. 

  There are no transformations applied to the set of features.

  Thus, feature selection selects a minimal subset of the variables that contain all predictive information necessary to produce a predictive model for the target variable (outcome).

  Examples: Backward elimination, Forward selection, and Random forests. 

- In **feature extraction**, we find a combination of new features and an appropriate transformation is applied to the set of features. 

  The new set of features contains different values rather than the original values. 

  Feature extraction can be further divided into _linear_ methods and _non-linear_ methods.


----------


## Feature Importance

[3 Essential Ways to Calculate Feature Importance in Python](https://towardsdatascience.com/3-essential-ways-to-calculate-feature-importance-in-python-2f9149592155)

### Dataset loading and preparation

### Method 1: Obtain importances from correlation coefficients

### Method 2: Obtain importances from a tree-based model

### Method 3: Obtain importances from PCA loading scores



## Feature Engineering Techniques


----------


## Transform Target Variables for Regression

Performing data preparation operations such as scaling is relatively straightforward for input variables and has been made routine in Python via the `Pipeline` scikit-learn class.

On regression predictive modeling problems where a numerical value must be predicted, it can also be crucial to scale and perform other data transformations on the target variable which can be achieved using the `TransformedTargetRegressor` class.

For regression problems, it is often desirable to scale or transform both the input and the target variables.

### Importance of Data Scaling

It is common to have data where the scale of values differs from variable to variable.

For example, one variable may be in feet, another in meters, etc.

Some machine learning algorithms perform much better if all of the variables are scaled to the same range, such as scaling all variables to values between 0 and 1 called normalization.

This effects algorithms that use a weighted sum of the input such as linear models and neural networks as well as models that use distance measures such as support vector machines and k-nearest neighbors.

Therefore, it is a good practice to scale input data and perhaps even try other data transforms such as making the data more normal (Gaussian probability distribution) using a power transform.

This also applies to output variables called _target_ variables such as numerical values that are predicted when modeling regression predictive modeling problems.

For regression problems, it is often desirable to scale or transform both the input and the target variables.

Scaling input variables is straightforward. In scikit-learn, you can use the scale objects manually or the more convenient `Pipeline` that allows you to chain a series of data transform objects together before using your model.

The `Pipeline` will fit the scale objects on the training data for you and apply the transform to new data, such as when using a model to make a prediction.

```py
    from sklearn.pipeline import Pipeline
    
    # prepare the model with input scaling
    pipeline = Pipeline(steps=[
        ('normalize', MinMaxScaler()), 
        ('model', LinearRegression())])
    
    # fit pipeline
    pipeline.fit(train_x, train_y)
    
    # make predictions
    yhat = pipeline.predict(test_x)
```

### How to Scale Target Variables?

There are two ways that you can scale target variables:

1. Manually transform the target variable.
2. Automatically transform the target variable.

Manually managing the scaling of the target variable involves creating and applying the scaling object to the data manually.

1. Create the transform object such as `MinMaxScaler`.
2. Fit the transform on the training dataset.
3. Apply the transform to the train and test datasets.
4. Invert the transform on any predictions made.

```py
    # create target scaler object
    target_scaler = MinMaxScaler()
    target_scaler.fit(train_y)

    # transform target variables
    train_y = target_scaler.transform(train_y)
    test_y = target_scaler.transform(test_y)

    # invert transform on predictions
    yhat = model.predict(test_X)
    yhat = target_scaler.inverse_transform(yhat)
```

However, if you use this approach then you cannot use convenience functions in scikit-learn such as `cross_val_score()` to quickly evaluate a model.

### Automatic Transform of the Target Variable

An alternate approach is to automatically manage the transform and inverse transform by using the `TransformedTargetRegressor` object that wraps a given model and a scaling object.

It will prepare the transform of the target variable using the same training data used to fit the model, then apply that inverse transform on any new data provided when calling predict(), returning predictions in the correct scale.

```py
    # define the target transform wrapper
    wrapped_model = TransformedTargetRegressor(regressor=model, transformer=MinMaxScaler())

    # use the target transform wrapper
    wrapped_model.fit(train_X, train_y)
    yhat = wrapped_model.predict(test_X)

    # use the target transform wrapper
    wrapped_model.fit(train_X, train_y)
    yhat = wrapped_model.predict(test_X)
```

This is much easier and allows us to use helper functions such as `cross_val_score()` to evaluate a model.


### Complete Regression Example

```py
    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.compose import TransformedTargetRegressor
 
    # load data
    dataset = np.loadtxt('housing.csv', delimiter=",")

    # split into inputs and outputs
    X, y = dataset[:, :-1], dataset[:, -1]
    
    # prepare the model with input scaling
    pipeline = Pipeline(steps=[('normalize', MinMaxScaler()), ('model', HuberRegressor())])
    
    # prepare the model with target scaling
    model = TransformedTargetRegressor(regressor=pipeline, transformer=MinMaxScaler())
    
    # evaluate model
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
    
    # convert scores to positive
    scores = np.absolute(scores)
    
    # summarize the result
    s_mean = np.mean(scores)
    print('Mean MAE: %.3f' % (s_mean))
```

We can also explore using other data transforms on the target variable such as the `PowerTransformer` to make each variable more Gaussian-like (using the Yeo-Johnson transform) and improve the performance of linear models.

By default, the `PowerTransformer` also performs a standardization of each variable after performing the transform.

```py
    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import HuberRegressor
    from sklearn.preprocessing import PowerTransformer
    from sklearn.compose import TransformedTargetRegressor

    # load data
    dataset = np.loadtxt('housing.csv', delimiter=",")

    # split into inputs and outputs
    X, y = dataset[:, :-1], dataset[:, -1]

    # prepare the model with input scaling
    pipeline = Pipeline(steps=[('power', PowerTransformer()), ('model', HuberRegressor())])

    # prepare the model with target scaling
    model = TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer())

    # evaluate model
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)

    # convert scores to positive
    scores = np.absolute(scores)

    # summarize the result
    s_mean = np.mean(scores)
    print('Mean MAE: %.3f' % (s_mean))
```



## References

[What is Feature Engineering?](https://towardsdatascience.com/what-is-feature-engineering-importance-tools-and-techniques-for-machine-learning-2080b0269f10?source=rss----7f60cf5620c9---4)

[How to Transform Target Variables for Regression in Python](https://machinelearningmastery.com/how-to-transform-target-variables-for-regression-with-scikit-learn/)


[Representation: Feature Engineering](https://developers.google.com/machine-learning/crash-course/representation/feature-engineering)

[Basic Feature Discovering for Machine Learning](https://medium.com/diko-hary-adhanto-portfolio/basic-feature-discovering-for-machine-learning-cbd47bf4b651)


[Techniques for Dimensionality Reduction](https://towardsdatascience.com/techniques-for-dimensionality-reduction-927a10135356)


### Time Series Data Preparation

[How to Normalize and Standardize Time Series Data in Python](https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/)

[4 Common Machine Learning Data Transforms for Time Series Forecasting](https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/)

[How to Scale Data for Long Short-Term Memory Networks in Python](https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/)

	



