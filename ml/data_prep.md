# Data Preparation

[Tour of Data Preparation Techniques for Machine Learning](https://machinelearningmastery.com/data-preparation-techniques-for-machine-learning/)

[Feature Engineering](https://gist.github.com/codecypher/dd4c7e8794982570288c2cfe95665c9c)

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


## Handle errors in variables

- Missing Data:  can happen due to forgotten to store, inappropriate data handling, inefficient data entry at the ground level, etc. 

- High Cardinality: the number of different labels in categorical data is very high, which causes problems to model to learn.

- Outliers: the extreme cases that may be due to error, but not in every case.


## Data Cleaning

[How to Perform Data Cleaning for Machine Learning with Python?](https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/)

1. Handling missing values
2. Scaling and normalization
3. Parsing dates
4. Character encodings
5. Inconsistent Data Entry


## Order of Data Transforms

You may want to experiment with applying multiple data transforms to a time series prior to modeling.

It is quite common to;

- apply a power transform to remove an increasing variance
- apply seasonal differencing to remove seasonality
- apply one-step differencing to remove a trend.

The order that the transform operations are applied is important.



----------

## Data Cleaning Challenge

[Kaggle Data Cleaning Challenge: Missing values](https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values)

The [Data Science Primer](https://elitedatascience.com/primer) covers exploratory analysis, data cleaning, feature engineering, algorithm selection, and model training.

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





## References

### Glossary

[ML Cheatsheet](https://github.com/shuaiw/ml-cheatsheet)
[ML Glossary](https://ml-cheatsheet.readthedocs.io/en/latest/index.html)
[Analytics Vidhya Glossary of Machine Learning Terms](https://www.analyticsvidhya.com/glossary-of-common-statistics-and-machine-learning-terms/#five)


### Data Preprocessing

[A Better Way for Data Preprocessing: Pandas Pipe](https://towardsdatascience.com/a-better-way-for-data-preprocessing-pandas-pipe-a08336a012bc)
[How to Select a Data Splitting Method](https://towardsdatascience.com/how-to-select-a-data-splitting-method-4cf6bc6991da)


### Exploratory Data Analysis (EDA)

[Reading and interpreting summary statistics](https://towardsdatascience.com/reading-and-interpreting-summary-statistics-df34f4e69ba6)
[11 Essential Code Blocks for Complete EDA (Exploratory Data Analysis)-Regression Task](https://towardsdatascience.com/11-simple-code-blocks-for-complete-exploratory-data-analysis-eda-67c2817f56cd)
[Python Cheat Sheet for Data Science](https://chipnetics.com/tutorials/python-cheat-sheet-for-data-science/)

