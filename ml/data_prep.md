# Data Preparation

[Tour of Data Preparation Techniques for Machine Learning](https://machinelearningmastery.com/data-preparation-techniques-for-machine-learning/)

[Feature Engineering](https://gist.github.com/codecypher/dd4c7e8794982570288c2cfe95665c9c)

<!-- MarkdownTOC -->

- Data Preparation
    - Import data
    - Format adjustments
    - Correct inconsistencies
    - Handle errors in variables
- Data Cleaning
    - Add Dummy Variables
    - Highly Imbalanced Data
    - Order of Data Transforms for Time Series
- Data Cleaning Challenge
- Scaling vs. Normalization
    - Scaling
    - Normalization
- Data Pipelines
    - Create a simple Pipeline
    - Best Scaler
    - Best Estimator
    - Pipeline with With PCA
    - Joblib
    - Pandas pipe
- References
    - Glossary
    - Data Preprocessing
    - Categorical Data
    - Exploratory Data Analysis \(EDA\)

<!-- /MarkdownTOC -->


## Exploratory Data Analysis (EDA)

[11 Essential Code Blocks for EDA Regression Task](https://towardsdatascience.com/11-simple-code-blocks-for-complete-exploratory-data-analysis-eda-67c2817f56cd)

Exploratory Data Analysis (EDA) is one of the first steps of the data science process which involves learning as much as possible about the data without spending too much time. 

### Summary Statistics

[Reading and interpreting summary statistics](https://towardsdatascience.com/reading-and-interpreting-summary-statistics-df34f4e69ba6)

It is important to know how to extract information from descriptive statistics. 


```py

# inspect the data 
Airlines.head()

# get the data info
Airlines.info()

# check the shape of the data-frame
Airlines.shape

# check for missing values
Airlines.isna()

# check for duplicate values
Airlines.duplicated() 
```


## Data Preparation

### Import data

[Read datasets with URL](https://towardsdatascience.com/dont-download-read-datasets-with-url-in-python-8245a5eaa919)

- Split data along delimiters (CSV)

- Extract parts from data entries (Do you only need a part of a certain attribute?)

- Remove leading and trailing spaces

### Format adjustments

- Standardize types (decimal separators, date formats, or measurement units)

- Replace unrecognizable or corrupted characters

- Check for truncated entries (data entries are cut off at a certain position)

### Correct inconsistencies

- Check for inconsistent entries (such as age cannot be negative) 

- Check for data outliers for numerical data

- Check for wrong categories for categorical data (imilar products are not put into different categories)

- Handle missing values (add data or remove rows)

- Handle/Remove duplicates

### Handle errors in variables

- Missing Data: can happen due to forgotten to store, inappropriate data handling, inefficient data entry at the ground level, etc. 

- High Cardinality: the number of different labels in categorical data is very high, which causes problems to model to learn.

- Outliers: the extreme cases that may be due to error, but not in every case.


## Data Cleaning

[How to Perform Data Cleaning for Machine Learning with Python?](https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/)

Data cleaning refers to identifying and correcting errors in the dataset that may negatively impact a predictive model.

- Identify Columns That Contain a Single Value

- Delete Columns That Contain a Single Value

- Consider Columns That Have Very Few Values

- Remove Columns That Have A Low Variance

- Identify Rows that Contain Duplicate Data

- Delete Rows that Contain Duplicate Data

1. Handling missing values
2. Scaling and normalization
3. Parsing dates
4. Inconsistent Data Entry

### Add Dummy Variables

[Ordinal and One-Hot Encodings for Categorical Data](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/)

Most machine learning algorithms cannot directly handle categorical features. Specifically, they cannot handle _text values_.

Therefore, we need to create dummy variables for our categorical features which is called _one-hot encoding_.

A **dummy variable** is a binary (0 or 1) variable that represents a single class from a categorical feature.

The information you represent is exactly the same but the numeric representation allows you to pass the values to be process by ML algorithms.

The one-hot encoding creates one binary variable for each category which includes redundancy. 

In contrast, a dummy variable encoding represents C categories with C-1 binary variables.

```py
    pd.get_dummies(df, columns=['Color'], prefix=['Color'])
```

### Highly Imbalanced Data

Need to upsample, but categories with only 1 entry when oversampled will give a 100% accuracy and artificially inflate the total accuracy/precision.

- We can use `UpSample` in Keras/PyTorch and `pd.resample()`


### Order of Data Transforms for Time Series

You may want to experiment with applying multiple data transforms to a time series prior to modeling.

It is quite common to;

- apply a power transform to remove an increasing variance

- apply seasonal differencing to remove seasonality

- apply one-step differencing to remove a trend.

The order that the transform operations are applied is important.


----------



## Data Cleaning

The [Data Science Primer](https://elitedatascience.com/primer) covers exploratory analysis, data cleaning, feature engineering, algorithm selection, and model training.

1. Handling missing values
2. Scaling and normalization
3. Parsing dates
4. Character encodings
5. Inconsistent Data Entry


## Scaling vs. Normalization

The process of scaling and normalization are very similar. In both cases, you are transforming the values of numeric variables so that the transformed data points have specific helpful properties.

- In scaling, you are changing the _range_ of your data. 
- In normalization, you are changing the _shape_ of the distribution of your data.


### Scaling

Some machine learning algorithms perform much better if all of the variables are scaled to the same range such as scaling all variables to values between 0 and 1 which is called normalization.

This effects algorithms that use a weighted sum of the input (such as linear models and neural networks) as well as models that use distance measures (such as support vector machines and k-nearest neighbors).

Therefore, it is a best practice to scale input data.

In scaling, you are transforming your data so that it fits within a specific scale such as 0-100 or 0-1.

It is common to have data where the scale of values differs from variable to variable. For example, one variable may be in feet and another in meters (pounds vs inches, kilograms vs meters).

By scaling your variables, you can help compare different variables on equal footing.

You especially want to scale data when you are using methods based on **measures of the distance between data points** such as support vector machines (SVM) or k-nearest neighbors (KNN). With these algorithms, a change of "1" in any numeric feature is given the same importance.


### Normalization

Scaling just changes the range of your data. 

The point of normalization is to change your observations so that they can be described as a _normal distribution_.

**Normal distribution:** This is a specific statistical distribution (bell curve) where a roughly equal number of observations fall above and below the mean, the mean and the median are the same, and there are more observations closer to the mean. The normal distribution is also known as the _Gaussian distribution_.

In general, you only want to normalize your data if you are going to be using a machine learning or statistics technique that assumes your data is normally distributed. 

Some examples are: t-tests, ANOVAs, linear regression, linear discriminant analysis (LDA), and Gaussian naive Bayes.

TIP: any method with "Gaussian" in the name probably assumes normality.

The method we are using to normalize here is called the Box-Cox Transformation.

Let us take a quick peek at what normalizing some data looks like:

Notice that the shape of our data has changed.


Before normalizing it was almost L-shaped but after normalizing it looks more like the outline of a bell (hence "bell curve").


## Data Pipelines

[Build Machine Learning Pipelines](https://medium.datadriveninvestor.com/build-machine-learning-pipelines-with-code-part-1-bd3ed7152124?gi=c419327a3c8c)

Pipeline is a technique used to create a linear sequence of data preparation and modeling steps to automate machine learning workflows.

Pipelines also help in parallelization which means different jobs can be run in parallel as well as help to inspect and debug the data flow in the model.


### Create a simple Pipeline

```py
    # Create a pipeline
    pipeline_lr = Pipeline([
        ('mms', MinMaxScaler()),
        ('lr', LogisticRegression())
    ])
    
    # Fit pipeline
    pipeline_lr.fit(trainX, trainY)
    
    # Evaluate pipeline
    y_predict = pipeline_lr.predict(testX)
    print('Test Accuracy Score: {:.4f}'.format(accuracy_score(testY, y_predict)))
```

### Best Scaler

```py
# Create a pipeline
pipeline_lr_mm = Pipeline([
    ('mms', MinMaxScaler()),
    ('lr', LogisticRegression())
    ])
pipeline_lr_r = Pipeline([
    ('rs', RobustScaler()),
    ('lr', LogisticRegression())
    ])
pipeline_lr_w = Pipeline([
    ('lr', LogisticRegression())
    ])
pipeline_lr_s = Pipeline([
    ('ss', StandardScaler()),
    ('lr', LogisticRegression())
    ])
    
# Create a pipeline dictionary
pipeline_dict = {
0: 'Logistic Regression without scaler',
    1: 'Logistic Regression with MinMaxScaler',
    2: 'Logistic Regression with RobustScaler',
    3: 'Logistic Regression with StandardScaler',
}

# Create a pipeline list
pipelines = [pipeline_lr_w, pipeline_lr_mm, 
    pipeline_lr_r, 
    pipeline_lr_s]

# Fit the pipeline
for p in pipelines:
    p.fit(trainX, trainY)

# Evaluate the pipeline
for i, val in enumerate(pipelines):
print('%s pipeline Test Accuracy Score: %.4f' % (pipeline_dict[i], accuracy_score(testY, val.predict(testX))))
```

Convert it to dataFrame and show the best model:

```py
l = []
for i, val in enumerate(pipelines):
    l.append(accuracy_score(testY, val.predict(testX)))
result_df = pd.DataFrame(list(pipeline_dict.items()),columns = ['Index','Estimator'])

result_df['Test_Accuracy'] = l

best_model_df = result_df.sort_values(by='Test_Accuracy', ascending=False)
print(best_model_df)
```

### Best Estimator

```py
# Create a pipeline
pipeline_knn = Pipeline([
    ('ss1', StandardScaler()),
    ('knn', KNeighborsClassifier(n_neighbors=4))
    ])
pipeline_dt = Pipeline([
    ('ss2', StandardScaler()),
    ('dt', DecisionTreeClassifier())
    ])
pipeline_rf = Pipeline([
    ('ss3', StandardScaler()),
    ('rf', RandomForestClassifier(n_estimators=80))
    ])
pipeline_lr = Pipeline([
    ('ss4', StandardScaler()),
    ('lr', LogisticRegression())
    ])
pipeline_svm_lin = Pipeline([
    ('ss5', StandardScaler()),
    ('svm_lin', SVC(kernel='linear'))
    ])
pipeline_svm_sig = Pipeline([
    ('ss6', StandardScaler()),
    ('svm_sig', SVC(kernel='sigmoid'))
    ])
    
# Create a pipeline dictionary
pipeline_dict = {
    0: 'knn',
    1: 'dt',
    2: 'rf',
    3: 'lr',
    4: 'svm_lin',
    5: 'svm_sig',
    
    }

# Create a List
pipelines = [pipeline_lr, pipeline_svm_lin, pipeline_svm_sig, pipeline_knn, pipeline_dt, pipeline_rf]

# Fit the pipeline
for p in pipelines:
    pipe.fit(trainX, trainY)

# Evaluate the pipeline
l = []
for i, val in enumerate(pipelines):
    l.append(accuracy_score(testY, val.predict(testX)))
    
result_df = pd.DataFrame(list(pipeline_dict.items()),columns = ['Idx','Estimator'])

result_df['Test_Accuracy'] = l

b_model = result_df.sort_values(by='Test_Accuracy', ascending=False)

print(b_model)
```

### Pipeline with With PCA

Pipeline example with Principal Component Analysis (PCA)


### Joblib

[Lightweight Pipelining In Python using Joblib](https://towardsdatascience.com/lightweight-pipelining-in-python-1c7a874794f4)

Joblib is an open-source Python library that helps in saving pipelines to a file which can be used later.

### Pandas pipe

The pandas `pipe` function offers a structured and organized way for combining several functions into a single operation.

As the number of steps increase, the syntax becomes cleaner with the pipe function compared to executing functions separately.

[A Better Way for Data Preprocessing: Pandas Pipe](https://towardsdatascience.com/a-better-way-for-data-preprocessing-pandas-pipe-a08336a012bc)


## Bootstrapping

The goal of bootstrap is to create an estimate (sample mean x̄) for a population parameter (population mean θ) based on multiple data samples obtained from the original sample.

Bootstrapping is done by repeatedly sampling (with replacement) the sample dataset to create many simulated samples. 

Each simulated bootstrap sample is used to calculate an estimate of the parameter and the estimates are then combined to form a sampling distribution.

The bootstrap sampling distribution then allows us to draw statistical inferences such as estimating the standard error of the parameter.



## References

### Glossary

[ML Cheatsheet](https://github.com/shuaiw/ml-cheatsheet)

[ML Glossary](https://ml-cheatsheet.readthedocs.io/en/latest/index.html)

[Analytics Vidhya Glossary of Machine Learning Terms](https://www.analyticsvidhya.com/glossary-of-common-statistics-and-machine-learning-terms/#five)


### Data Preprocessing

[Data Science Primer](https://elitedatascience.com/primer)

[A Better Way for Data Preprocessing: Pandas Pipe](https://towardsdatascience.com/a-better-way-for-data-preprocessing-pandas-pipe-a08336a012bc)

[How to Select a Data Splitting Method](https://towardsdatascience.com/how-to-select-a-data-splitting-method-4cf6bc6991da)

[Kaggle Data Cleaning Challenge: Missing values](https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values)


### Categorical Data

[Smarter Ways to Encode Categorical Data for Machine Learning](https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159)

[Stop One-Hot Encoding Your Categorical Variables](https://towardsdatascience.com/stop-one-hot-encoding-your-categorical-variables-bbb0fba89809)


### Exploratory Data Analysis (EDA)

[Reading and interpreting summary statistics](https://towardsdatascience.com/reading-and-interpreting-summary-statistics-df34f4e69ba6)

[11 Essential Code Blocks for Complete EDA (Exploratory Data Analysis) Regression Task](https://towardsdatascience.com/11-simple-code-blocks-for-complete-exploratory-data-analysis-eda-67c2817f56cd)

[Python Cheat Sheet for Data Science](https://chipnetics.com/tutorials/python-cheat-sheet-for-data-science/)

