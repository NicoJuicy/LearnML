# Exploratory Data Analysis (EDA)

## Exploratory Data Analysis Example

[How to build a Machine Learning (ML) based Predictive System](https://towardsdatascience.com/machine-learning-ml-based-predictive-system-to-predict-the-satisfaction-level-of-airlines-f0780dbdbc87?source=rss----7f60cf5620c9---4)

Check the summary statistics and create histograms for the numeric variables of the dataset as presented in the code below.

```py
    numeric_variables = list(df.select_dtypes(include=['int64', 'float64'])) #select the numeric variables

    df[numeric_variables].describe().apply(lambda x:round(x,2)).T #apply describe method

    histograms = df[numeric_variables].hist(bins =10, xlabelsize=10, ylabelsize=10, grid=False, sharey= True, figsize = (15,15)) #create the histograms
```

Study the relationship between satisfaction and class category.

Investigate the relationship between total delay time, overall rating, and satisfaction. 

Check if age is affecting the satisfaction of customers. 

## Summary Statistics

[Reading and interpreting summary statistics](https://towardsdatascience.com/reading-and-interpreting-summary-statistics-df34f4e69ba6)

It is important to know how to extract information from descriptive statistics. 

### Statistical Distribution

#### Mean

With the mean value, you are trying to get a sense of what an average data point looks like

#### Standard Deviation

Standard deviation is a measure of variation/dispersion of data points with respect to the mean.

Smaller STD indicates that the data are mostly centered around the mean whereas a higher STD value indicates the data points are rather dispersed.

#### Median (50%)

The 50th percentile (the 50% column) is also known as the median. Like mean, it’s another measure of central tendency.

Median is a preferred metric rather than mean if there are outliers or high variability in the data.

If the difference between mean and median is _small_, you can infer that the data is symmetrically distributed.

If the median is higher than the mean, data is likely left-skewed in distribution.

#### Min and Max

Min and max values represent the lower and upper limit of a variable in the dataset.


### Anomalies

You can get a sense of outliers, anomalies, and other points of interest in the dataset using descriptive statistics.

#### Outliers

A large difference between the 75th percentile and the maximum value indicates the presence of potential outliers.

Likewise, a large difference between the minimum value and the 25th percentile indicates the presence of potential outliers.

To confirm outliers you can create a boxplot for visual inspection:

```py
    sns.boxplot(y=df['total_bill']);
```

#### Red flags

Sometimes descriptive statistics can raise red flags.

Places with unexpected minimum values (0 or negative) or absolutely unacceptible maximum values (such as someone’s age 120 years!).

These are obvious indications that there are issues in the data and need further investigation.


## Describing categorical data

So far we have investigated descriptive statistics for numeric variables. Python pandas also offer a summary for categorical variables.

```py
    df.desribe(include='category')
```


## Essential Code Blocks

[11 Essential Code Blocks for EDA Regression Task](https://towardsdatascience.com/11-simple-code-blocks-for-complete-exploratory-data-analysis-eda-67c2817f56cd)

Exploratory Data Analysis (EDA) is one of the first steps of the data science process which involves learning as much as possible about the data without spending too much time. 

We can get an instinctive as well as a high-level practical understanding of the data including a general idea of the structure of the data set, some cleaning ideas, the target variable and possible modeling techniques.

## Basic Dataset Exploration

1. Shape (dimensions) of the DataFrame
2. Data types of the various columns

   We may observe that our dataset has a combination of categorical (object) and numeric (float and int) features.

   Look for: Numeric features that should be categorical and vice versa.

3. Display a few rows

## Summary Statistics

```py
    # numerical features
    df.describe()

    # include categorical features
    df.describe(include=['object', 'category'])
    # unique = number of unique categories
    # top = dominant category
    # freq = count of dominant category
```

## Plot of numeric features

```py
    df.hist(figsize=(14,14), xrot=45)
    plt.show()
```

What to look for:

- Possible outliers that cannot be explained or might be measurement errors. 

- Numeric features that should be categorical such as Gender represented by 1 and 0.

- Boundaries that do not make sense such as percentage values> 100.

## Plot of categorical features

```py
    for column in df.select_dtypes(include='object'):
        if df[column].nunique() < 10:
            sns.countplot(y=column, data=data)
    plt.show()
```

What to look for:

- Sparse classes which have the potential to affect model performance.
- Mistakes in labeling of the classes such as 2 classes with minor spelling differences.

## Segment the target variable by categorical features

Compare the _target_ feature (Price) between the various classes of our main categorical features (Type, Method, and Regionname) and see how the target changes with the classes.

```py
    for column in data.select_dtypes(include=’object’):
        if data[column].nunique() < 10:
            sns.boxplot(y=column, x=’Price’, data=data)
    plt.show()
```

What to look for: Classes that most affect the target variable(s).

## Group numeric features by each categorical feature

See how all the other numeric features (not just target feature) change with each categorical feature by summarizing the numeric features across the classes.

```py
    for column in data.select_dtypes(include='object'):
        if data[column].nunique() < 10:
            display(data.groupby(column).mean())
```

## Relationships between numeric features

### Correlation matrix for numerical features

A _correlation_ is a value between -1 and 1 that amounts to how closely values of two separate features move simultaneously.

A _positive_ correlation means that as one feature increases the other one also increases while a _negative_ correlation means one feature increases as the other decreases.

Correlations close to 0 indicate a _weak_ relationship while closer to -1 or 1 signifies a _strong_ relationship.

```py
    corrs = df.corr()
    print(corrs)
```

This might not mean much now, so we can plot a **heatmap** to visualize the correlations.

### Heatmap of the correlations

```py
    plt.figure(figsize=(10, 8))
    sns.heatmap(corrs, cmap='RdBu_r', annot=True)
    plt.show()
```

What to look for:

- Strongly correlated features; either dark red (positive) or dark blue(negative).
- Target variable: If it has strong positive or negative relationship with other features.


----------


# A Practical Guide to Linear Regression

Linear regression is a typical regression algorithm that is responsible for numerous prediction. 

In a nutshell, a linear regression finds the optimal linear relationship between independent variables and dependent variables, then makes prediction accordingly.

## Exploratory Data Analysis (EDA)

EDA is essential to both investigate the data quality and reveal hidden correlations among variables.

1. Univariate Analysis

Visualize the data distribution using histogram for numeric variables and bar chart for categorical variables.

### Why do we need univariate analysis?

- Determine if dataset contains outliers

- Detrmine if we need data transformations or feature engineering

In this case, we found out that “expenses” follows a power law distribution, which means that log transformation is required as a step of feature engineering step, to convert it to normal distribution.

2. Multivariate Analysis

When thinking of linear regression, the first visualization technique that we can think of is scatterplot. 

By plotting the target variable against the independent variables using a single line of code `sns.pairplot(df)`, the underlying linear relationship becomes more evident.

3. Correlation Analysis

Correlation analysis examines the linear correlation between variable pairs which can be achieved by combining `corr()` function with `sns.heatmap()`. 

### Why do we need correlation analysis?

- To identify collinearity between independent variables — linear regression assumes no collinearity among independent features, so it is essential to drop some features if collinearity exists. 

- To identify independent variables that are strongly correlated with the target — strong predictors.

## Feature Engineering

EDA brought some insights of what types of feature engineering techniques are suitable for the dataset.

1. Log Transformation

We  found that the target variable  “expenses” is right skewed and follows a power law distribution. 

Since linear regression assumes linear relationship between input and output variable, it is necessary to use log transformation to “expenses” variable. 

As shown below, the data tends to be more normally distributed after applying `np.log2()`.

2. Encoding Categorical Variable

Another requirement of machine learning algorithms is to encode categorical variable into numbers.

Two common methods are one-hot encoding and label encoding. 


## Model Implementation

A simple linear regression y = b0 + b1x predicts relationship between one independent variable x and one dependent variable y. 

As more features/independent variables are introduced, it becomes multiple linear regression y = b0 + b1x1 + b2x2 + … + bnxn, which cannot be easily plotted using a line in a two dimensional space.

Here we use `LinearRegression()` class from scikit-learn to implement the linear regression. 

We specify `normalize = True` so that independent variables will be normalized and transformed into same scale. 

Note that scikit-learn linear regression utilizes **Ordinary Least Squares** to find the optimal line to fit the data which means the line, defined by coefficients b0, b1, b2 … bn, minimizes the residual sum of squares between the observed targets and the predictions (the blue lines in chart). 


## Model Evaluation

Linear regression model can be qualitatively evaluated by visualizing error distribution. 

There are also quantitative measures such as MAE, MSE, RMSE and R squared.

1. Error Distribution

Here we use a histogram to visualize the distribution of error which should somewhat conform to a normal distribution. 

A non-normal error distribution may indicates that there is non-linear relationship that model failed to pick up or more data transformations are needed.

2. MAE, MSE, RMSE

All three methods measure the errors by calculating the difference between predicted values ŷ and actual value y, so the smaller the better. 

The main difference is that MSE/RMSE penalized large errors and are differentiable whereas MAE is not differentiable which makes it hard to apply in gradient descent. 

Compared to MSE, RMSE takes the square root which maintains the original data scale.

3. R Squared

R squared or _coefficient of determination_ is a value between 0 and 1 that indicates the amount of variance in actual target variables explained by the model. 

R squared is defined as 1 — RSS/TSS which is 1 minus the ratio between sum of squares of residuals (RSS) and total sum of squares (TSS). 

Higher R squared means better model performance.

In this case, a R squared value of 0.78 indicating that the model explains 78% of variation in target variable, which is generally considered as a good rate and not reaching the level of overfitting.


----------



# Essential Code Blocks

[11 Essential Code Blocks for Complete EDA](https://towardsdatascience.com/11-simple-code-blocks-for-complete-exploratory-data-analysis-eda-67c2817f56cd)

## Basic data set Exploration

1. Shape (dimensions) of the DataFrame

2. Data types of the various columns

What to look out for;

- Numeric features that should  be categorical and vice versa.

3. Display a few rows

What to look out for:

- Can you understand the column names? Do they make sense? (Check the variable definitions again if needed)
- Do the values in these columns make sense?
- Are there significant missing values (NaN)?
- What types of classes do the categorical features have?

## Distribution

This refers to how the values in a feature are distributed or how often they occur.

For numeric features, we will see how many times groups of numbers appear in a particular column.

For categorical features, the classes for each column and their frequency.

We will use both graphs and actual summary statistics.

The graphs enable us to get an overall idea of the distributions while the statistics give us factual numbers.

Both graphs and statistics are recommended since they complement each other.

### Numeric Features

4. Plot each numeric feature

```py
    df.hist(figsize=(14,14), xrot=45)
    plt.show()
```

What to look out for:

- Possible outliers that cannot be explained or might be measurement errors
- Numeric features that should be categorical. For example, Gender represented by 1 and 0.
- Boundaries that do not make sense such as percentage values> 100.

5. Summary statistics of the numerical features

```py
    print(df.describe())

    # count null values
    print(df.isnull().sum())

    # count of students whose physics marks are greater than 11
    df[df['Physics'] > 11]['Name'].count())

    # count students whose physics marks are greater than 10
    # and math marks are greater than 9.
    df[(df['Physics'] > 10 ) &
       (df['Math'] > 9 )]

   # Multi-column frequency count
   count = df.groupby(col_name).count()
   print(count)
  ```

We can see for each numeric feature, the count of valueS, the mean value, std or standard deviation, minimum value, the 25th percentile, the 50th percentile or median, the 75th percentile, and the maximum value.

What to look out for:

- Missing values: their count is not equal to the total number of rows of the dataset.
- Minimum or maximum values they do not make sense.
- Large range in values (min/max)


6. Summary statistics of the categorical features

```py
    df.describe(include=’object’)
```

7. Plot each categorical feature

```py
for column in df.select_dtypes(include='object'):
    if df[column].nunique() < 10:
        sns.countplot(y=column, data=df)
        plt.show()
```

What to look out for:

- Sparse classes which have the potential to affect a model’s performance.
- Mistakes in labeling of the classes, for example 2 exact classes with minor spelling differences.

## Grouping and segmentation

Segmentation allows us to cut the data and observe the relationship between categorical and numeric features.

8. Segment the target variable by categorical features.

We will compare the target feature (Price) between the various classes of our main categorical features (Type, Method, and Regionname) and see how the Price changes with the classes.

```py
    # Plot boxplot of each categorical feature with Price.
    for column in df.select_dtypes(include=’object’):
        if df[column].nunique() < 10:
            sns.boxplot(y=column, x=’Price’, data=df)
            plt.show()
```

What to look out for:

- which classes most affect the target variables.


9. Group numeric features by each categorical feature.

Here we will see how all the other numeric features (not just Price) change with each categorical feature by summarizing the numeric features across the classes.

```py
    # For the 3 categorical features with less than 10 classes,
    # we group the data, then calculate the mean across the numeric features.
    for column in df.select_dtypes(include='object'):
        if df[column].nunique() < 10:
            display(df.groupby(column).mean())
```

## Relationships between numeric features and other numeric features

10. Correlations matrix for the different numerical features

A correlation is a value between -1 and 1 that amounts to how closely values of two separate features move simultaneously.

A positive correlation means that as one feature increases the other one also increases and a negative correlation means one feature increases as the other decreases.

Correlations close to 0 indicate a weak relationship while closer to -1 or 1 signifies a strong relationship.

```py
    corrs = data.corr()
    print(corrs)
```

This might not mean much now, so let us plot a heatmap to visualize the correlations.

11. Heatmap of the correlations


```py
    # Plot the grid as a rectangular color-coded matrix using seaborn heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(corrs, cmap='RdBu_r', annot=True)
    plt.show()
```

What to look out for:

- Strongly correlated features -- either dark red (positive) or dark blue(negative).

- Target variable: if it has strong positive or negative relationships with other features.



