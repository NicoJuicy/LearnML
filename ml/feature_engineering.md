# Feature Engineering

<!-- MarkdownTOC -->

- Feature Engineering Techniques
    - Imputation \(missing values\)
    - Handling Outliers
    - Log Transform
    - One-hot encoding
    - Scaling
- Normalization Techniques
    - Using maximum absolute scaling
    - Using min-max scaling
    - Using z-score scaling
- Transform Target Variables for Regression
    - Importance of Data Scaling
    - How to Scale Target Variables?
    - Automatic Transform of the Target Variable
    - Complete Regression Example
- Encoding Categorical Features
        - Complete One-Hot Encoding Example
        - Dummy Variable Encoding
- Common Questions on Normalization
    - Should I Normalize or Standardize?
    - Should I Standardize then Normalize?
    - Which Scaling Technique is Best?
    - How Do I Handle Out-of-Bounds Values?
- Common Questions on Encoding?
    - What if I have a mixture of numeric and categorical data?
    - What if I have hundreds of categories?
    - What encoding technique is the best?
- Feature Importance
    - Dataset loading and preparation
    - Method 1: Obtain importances from correlation coefficients
    - Method 2: Obtain importances from a tree-based model
    - Method 3: Obtain importances from PCA loading scores
- References
    - Categorical Data
    - Dimensionality Reduction
    - Feature Engineering
    - Scaling
    - Time Series Data Preparation

<!-- /MarkdownTOC -->


[What is Feature Engineering?](https://towardsdatascience.com/what-is-feature-engineering-importance-tools-and-techniques-for-machine-learning-2080b0269f10?source=rss----7f60cf5620c9---4)
[Data Preparation: Tips and Tricks](https://gist.github.com/codecypher/b8c85752acf287de28f816d9b9d75d08)

Feature engineering techniques for machine learning are a fundamental topic in machine learning, yet one that is often overlooked or deceptively simple.

Feature engineering consists of various processes:

- **Exploratory Data Analysis:** Exploratory data analysis (EDA) is a powerful and simple tool that can be used to improve your understanding of your data by exploring its properties. 

  EDA is often applied when the goal is to create hypotheses or find patterns in the data. 

  EDA is often used on large amounts of qualitative or quantitative data that haven’t been analyzed before.

- **Transformations:** Feature transformation is simply a function that transforms features from one representation to another. 

  The goal here is to plot and visualise data, if something is not adding up with the new features we can reduce the number of features used, speed up training, or increase the accuracy of a certain model.

- **Feature Creation:** Creating features involves creating new variables which will be most helpful for our model. 

  This can be adding or removing some features. As we saw above, the cost per sq. ft column was a feature creation.

- **Feature Extraction:** Feature extraction is the process of extracting features from a data set to identify useful information. 

  Without distorting the original relationships or significant information, this compresses the amount of data into manageable quantities for algorithms to process.

- **Benchmark:** A Benchmark Model is the most user-friendly, dependable, transparent, and interpretable model against which you can measure your own. 

  It is a good idea to run test datasets to see if your new machine learning model outperforms a recognised benchmark. These benchmarks are often used as measures for comparing the performance between different machine learning models


----------


## Feature Engineering Techniques

### 1. Imputation (missing values)

- Numerical Imputation
- Categorical Imputation

  Also see **Data Preparation**

### 2. Handling Outliers

- Removal: Outlier entries are deleted from the distribution

- Replacing: The outliers could be handled as missing values and replaced with suitable imputation.

- Capping: Using an arbitrary value or a value from a variable distribution to replace the maximum and minimum values.

- Discretization : Converting continuous variables into discrete values. 


### 3. Log Transform

Log Transform is the most used technique among data scientists to turn a skewed distribution into a normal or less-skewed distribution. 

We take the log of the values in a column and utilize those values as the column in this transform. 

Log transform is used to handle confusing data so that the data becomes more approximative to normal applications.


### 4. One-hot encoding

A one-hot encoding is a type of encoding in which an element of a finite set is represented by the index in that set where only one element has its index set to “1” and all other elements are assigned indices within the range [0, n-1]. 

In contrast to binary encoding schemes where each bit can represent 2 values (0 and 1), this scheme assigns a unique value for each possible case.


### 5. Scaling

Feature scaling is one of the most pervasive and difficult problems in machine learning, but it is one of the most important things to get right. 

In order to train a predictive model, we need data with a known set of features that needs to be scaled up or down as appropriate. 

- **Normalization:** All values are scaled in a specified range between 0 and 1 via normalisation (or min-max normalisation). 

  This modification has no influence on the feature’s distribution, however it does exacerbate the effects of outliers due to lower standard deviations. This, outliers should be dealt with prior to normalisation.

- **Standardization:** Standardization (z-score normalisation) is the process of scaling values while accounting for standard deviation. 

  If the standard deviation of features differs, the range of those features will likewise differ. 

  As a result, the effect of outliers in the characteristics is reduced. 

  To arrive at a distribution with a 0 mean and 1 variance, all the data points are subtracted by their mean and the result divided by the distribution’s variance.



## Normalization Techniques

[How to Use StandardScaler and MinMaxScaler Transforms in Python](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/)

[How to Use Power Transforms for Machine Learning](https://machinelearningmastery.com/power-transforms-with-scikit-learn/)

Data Normalization is a typical practice in machine learning which consists of transforming numeric columns to a _standard scale_. Some feature values may differ from others multiple times. Therefore, the features with higher values will dominate the learning process.

### Using maximum absolute scaling

The _maximum absolute_ scaling rescales each feature between -1 and 1 by dividing every observation by its maximum absolute value. 

We can apply the maximum absolute scaling in Pandas using the `.max()` and `.abs()` methods.

```py
    # copy the data
    df_max_scaled = df.copy()
      
    # apply normalization from scratch
    for column in df_max_scaled.columns:
        df_max_scaled[column] = df_max_scaled[column]  / df_max_scaled[column].abs().max()
```

### Using min-max scaling

The _min-max_ scaling (normalization) rescales the feature to the range of [0, 1] by subtracting the minimum value of the feature then dividing by the range. 

We can use `MinMaxScaler` class from sklearn.

```py
    from sklearn.preprocessing import MinMaxScaler

    # define scaler
    scaler = MinMaxScaler()

    # transform data
    scaled = scaler.fit_transform(data)
```

We can apply the min-max scaling in Pandas using the `.min()` and `.max()` methods which preserves the column headers/names.

```py
    # copy the data
    df_min_max_scaled = df.copy()
      
    # apply normalization from scratch
    for column in df_min_max_scaled.columns:
        df_min_max_scaled[column] = (df_min_max_scaled[column] - df_min_max_scaled[column].min()) / (df_min_max_scaled[column].max() - df_min_max_scaled[column].min())    
```


### Using z-score scaling

The _z-score_ scaling (standardization) transforms the data into a **normal (Gaussian) distribution** with a mean of 0 and a typical deviation of 1. Each standardized value is computed by subtracting the mean of the corresponding feature then dividing by the quality deviation.

We can apply standardization using `StandardScaler` class from sklearn.

```py
    from sklearn.preprocessing import StandardScaler

    # define scaler
    scaler = StandardScaler()

    # transform data
    scaled = scaler.fit_transform(data)
```

We can apply the standardization in Pandas using the `.min()` and `.max()` methods which preserves the column headers/names.

```py
    # copy the data
    df_z_scaled = df.copy()
      
    # apply normalization from scratch
    for column in df_z_scaled.columns:
        df_z_scaled[column] = (df_z_scaled[column] - df_z_scaled[column].mean()) / df_z_scaled[column].std()    
```


## Transform Target Variables for Regression

[How to Transform Target Variables for Regression in Python](https://machinelearningmastery.com/how-to-transform-target-variables-for-regression-with-scikit-learn/)

Performing data preparation operations such as scaling is relatively straightforward for input variables and has been made routine in Python via the `Pipeline` scikit-learn class.

On regression predictive modeling problems where a numerical value must be predicted, it can also be crucial to scale and perform other data transformations on the target variable which can be achieved in Python using the `TransformedTargetRegressor` class.

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

1. Create the transform object, e.g. a MinMaxScaler.
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


## Encoding Categorical Features

[Ordinal and One-Hot Encodings for Categorical Data](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/)

[3 Ways to Encode Categorical Variables for Deep Learning](https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/)

Machine learning algorithms and deep learning neural networks require that input and output variables are numbers.

This means that categorical data must be encoded to numbers before we can use it to fit and evaluate a model.

There are many ways to encode categorical variables for modeling, here are some of the most common:

  1. Integer (Ordinal) Encoding: each unique label/category is mapped to an integer.
  2. One Hot Encoding: each label is mapped to a binary vector.
  3. Dummy Variable Encoding
  4. Learned Embedding: a distributed representation of the categories is learned.


#### Complete One-Hot Encoding Example

A one-hot encoding is appropriate for categorical data where no relationship exists between categories.

The scikit-learn library provides the OneHotEncoder class to automatically one hot encode one or more variables.

By default the `OneHotEncoder` class will output data with a sparse representation which is efficient because most values are 0 in the encoded representation. However, we can disable this feature by setting the `sparse=False` so that we can review the effect of the encoding.

```py
    import numpy as np
    import pandas as pd

    from numpy import mean
    from numpy import std
    from pandas import read_csv

    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.metrics import accuracy_score

    # define the location of the dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer.csv"

    # load the dataset
    dataset = read_csv(url, header=None)

    # retrieve the array of data
    data = dataset.values

    # separate into input and output columns
    X = data[:, :-1].astype(str)
    y = data[:, -1].astype(str)

    # split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

    # one-hot encode input variables
    onehot_encoder = OneHotEncoder()
    onehot_encoder.fit(X_train)
    X_train = onehot_encoder.transform(X_train)
    X_test = onehot_encoder.transform(X_test)

    # ordinal encode target variable
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    y_test = label_encoder.transform(y_test)

    # define the model
    model = LogisticRegression()

    # fit on the training set
    model.fit(X_train, y_train)

    # predict on test set
    yhat = model.predict(X_test)

    # evaluate predictions
    accuracy = accuracy_score(y_test, yhat)
    print('Accuracy: %.2f' % (accuracy * 100))
```

#### Dummy Variable Encoding

The one-hot encoding creates one binary variable for each category.

The problem with one-hot encoding is that the representation includes redundancy. 

If we know that [1, 0, 0] represents “blue” and [0, 1, 0] represents “green” we do not need another binary variable to represent “red“. We could use 0 values for both “blue” and “green” alone: [0, 0].

This is called a dummy variable encoding and always represents C categories with C-1 binary variables.

In addition to being slightly less redundant, a dummy variable representation is required for some models such as linear regression model (and other regression models that have a bias term) since a one hot encoding will cause the matrix of input data to become singular which means it cannot be inverted, so the linear regression coefficients cannot be calculated using linear algebra. Therefore, a dummy variable encoding must be used.

However, we rarely encounter this problem in practice when evaluating machine learning algorithms other than linear regression.

It turns out that we can also use the `OneHotEncoder` class to implement a dummy encoding.


----------


## Common Questions on Normalization

This section lists some common questions and answers when scaling numerical data.

### Should I Normalize or Standardize?

Whether input variables require scaling depends on the specifics of your problem and of each variable.

You may have a sequence of quantities as inputs such as prices or temperatures.

If the distribution of the quantity is normal, it should be standardized. Otherwise, the data should be normalized 

The data should be normalized whether the range of quantity values is large (10s, 100s, ...) or small (0.01, 0.0001, ...).

If the quantity values are small (near 0-1) and the distribution is limited (such as standard deviation near 1) you might be able to get away with no scaling of the data.

Predictive modeling problems can be complex and it may not be clear how to best scale input data.

If in doubt, normalize the input sequence. If you have the resources, explore modeling with the raw data, standardized data, and normalized data and see if there is a beneficial difference in the performance of the resulting model.

### Should I Standardize then Normalize?

Standardization can give values that are both positive and negative centered around zero.

It may be desirable to normalize data after it has been standardized.

This might be a good approach if you have a mixture of standardized and normalized variables and would like all input variables to have the same minimum and maximum values as input for a given algorithm such as an algorithm that calculates distance measures.

### Which Scaling Technique is Best?

This is impossible to answer. The best approach would be to evaluate models on data prepared with each transform and use the transform or combination of transforms that result in the best performance for your data set and model.

### How Do I Handle Out-of-Bounds Values?

You may normalize your data by calculating the minimum and maximum on the training data.

Later, you may have new data with values smaller or larger than the minimum or maximum respectively.

One simple approach to handling this may be to check for such out-of-bound values and change their values to the known minimum or maximum prior to scaling. Alternately, you may want to estimate the minimum and maximum values used in the normalization manually based on domain knowledge.


## Common Questions on Encoding?

This section lists some common questions and answers when encoding categorical data.

### What if I have a mixture of numeric and categorical data?

What if I have a mixture of categorical and ordinal data?

You will need to prepare or encode each variable (column) in your dataset separately, then concatenate all of the prepared variables back together into a single array for fitting or evaluating the model.

Alternately, you can use the ColumnTransformer to conditionally apply different data transforms to different input variables.

### What if I have hundreds of categories?

What if I concatenate many one-hot encoded vectors to create a many-thousand-element input vector?

You can use a one-hot encoding up to thousands and tens of thousands of categories. Also, having large vectors as input sounds intimidating, but the models can generally handle it.

### What encoding technique is the best?

This is impossible to answer. The best approach would be to test each technique on your dataset with your chosen model and discover what works best.


----------


## Feature Importance

[3 Essential Ways to Calculate Feature Importance in Python](https://towardsdatascience.com/3-essential-ways-to-calculate-feature-importance-in-python-2f9149592155)

### Dataset loading and preparation

### Method 1: Obtain importances from correlation coefficients

### Method 2: Obtain importances from a tree-based model

### Method 3: Obtain importances from PCA loading scores


## When standardization or normalization should be used

Data-centric heuristics include the following:

1. If your data has outliers, use standardization or robust scaling.

2. If your data has a gaussian distribution, use standardization.

3. If your data has a non-normal distribution, use normalization.

Model-centric rules include these:

1. If your modeling algorithm assumes (but does not require) a normal distribution of the residuals (i.e., regularized linear regression, regularized logistic regression, or linear discriminant analysis), use standardization.

2. If your modeling algorithm makes no assumptions about the distribution of the data (i.e., k-nearest neighbors, support vector machines, and artificial neural networks), then use normalization.

In each use case, the rule proposes a mathematical fit with either the data or the learning model. 


## References

### Categorical Data

[Ordinal and One-Hot Encodings for Categorical Data](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/)

[Smarter Ways to Encode Categorical Data for Machine Learning](https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159)

[Stop One-Hot Encoding Your Categorical Variables](https://towardsdatascience.com/stop-one-hot-encoding-your-categorical-variables-bbb0fba89809)

### Dimensionality Reduction

[Techniques for Dimensionality Reduction](https://towardsdatascience.com/techniques-for-dimensionality-reduction-927a10135356)

### Feature Engineering

[Representation: Feature Engineering](https://developers.google.com/machine-learning/crash-course/representation/feature-engineering)

[Basic Feature Discovering for Machine Learning](https://medium.com/diko-hary-adhanto-portfolio/basic-feature-discovering-for-machine-learning-cbd47bf4b651)

### Scaling

[How to Selectively Scale Numerical Input Variables for Machine Learning](https://machinelearningmastery.com/selectively-scale-numerical-input-variables-for-machine-learning/)

[How to use Data Scaling Improve Deep Learning Model Stability and Performance](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/)

[How to Transform Target Variables for Regression in Python](https://machinelearningmastery.com/how-to-transform-target-variables-for-regression-with-scikit-learn/)

[The Mystery of Feature Scaling is Finally Solved](https://towardsdatascience.com/the-mystery-of-feature-scaling-is-finally-solved-29a7bb58efc2?source=rss----7f60cf5620c9---4)

### Time Series Data Preparation

[How to Normalize and Standardize Time Series Data in Python](https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/)

[4 Common Machine Learning Data Transforms for Time Series Forecasting](https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/)

[How to Scale Data for Long Short-Term Memory Networks in Python](https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/)

	



