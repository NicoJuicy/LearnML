# Data Preparation

<!-- MarkdownTOC levels=1,2,3 -->

- Overview
- Data Preparation
    - Import data
    - Format adjustments
    - Correct inconsistencies
    - Handle errors in variables
- Data Cleaning
    - Check data types
    - Handle missing values
    - Handle duplicate values
    - Handle Outliers
- Encoding Categorical Features
    - Integer \(Ordinal\) Encoding
    - Encoding Class Labels
    - One-Hot Encoding of Nominal Features
    - Dummy Variable Encoding
    - Complete One-Hot Encoding Example
- Scaling vs Normalization
    - Scaling
    - Normalization
    - Normalization vs Standardization
    - How to Choose between Standardization vs Normalization
    - Log Transform
- Normalization Techniques
    - Using maximum absolute scaling
    - Using min-max scaling
    - Using z-score scaling
- Parsing dates
- Inconsistent Data Entry
- Highly Imbalanced Data
- Order of Data Transforms for Time Series
- Train-Test Split
- Data Pipelines
    - Import Libraries
    - Create Simple Pipeline
    - Best Scaler
    - Best Estimator
    - Pipeline with PCA
- More on scikit-learn pipeline
    - Data Prep
    - Create Pipeline
    - Train and Evaluate Pipeline
    - Joblib
    - Pandas pipe
- Bootstrapping
- Code Examples and References
    - Data Preparation
    - Categorical Data
    - Scaling
    - Normalization
    - Train-Test Split
- References

<!-- /MarkdownTOC -->

## Overview

> Data quality is important to creating a successful machine learning model

[Feature Engineering](./feature_engineering.md)

> If you torture the data, it will confess to anything - Ronald Coase

There is often more than one way to sample and interogate data which means that deciding the best approach is subjective and open to bias. Thus, _data torture_ is the practice of repeatedly interpreting source data until it reveals a desired result. 


The approaches to data preparation will depend on the dataset as well as the data types, so there is no perfect list of steps.

[Concatenating CSV files using Pandas module](https://www.geeksforgeeks.org)

You try to handle the "worst" cases and not necessarily every case.

Exploratory Data Analysis (EDA) is crucial: summary stats and making plots of the data.

NOTE: It is estimated that 80% of AI project development time is spent on preparing the data [1]. 



## Data Preparation

The [Data Science Primer](https://elitedatascience.com/primer) covers exploratory analysis, data cleaning, feature engineering, algorithm selection, and model training.

7.1 Handling Missing Data
7.2 Data Transformation
7.3 String Manipulation

8.2 Combining and Merging Datasets
8.3 Reshaping and Pivoting

10. Data Aggregation and Group Operations


### Import data

[Read datasets with URL](https://towardsdatascience.com/dont-download-read-datasets-with-url-in-python-8245a5eaa919)

[13 ways to access data in Python](https://towardsdatascience.com/13-ways-to-access-data-in-python-bac5683e0063)

- Split data along delimiters (CSV)

- Extract parts from data entries (Do you only need part of a certain attribute?)

- Remove leading and trailing spaces

### Format adjustments

- Standardize types (decimal separators, date formats, or measurement units)

- Replace unrecognizable or corrupted characters

- Check for truncated entries (data entries that are cut off at a certain position)

### Correct inconsistencies

- Check for inconsistent entries (such as age cannot be negative) 

- Check for data outliers for numerical data

- Check for wrong categories for categorical data (imilar products are not put into different categories)

- Handle missing values (add data or remove rows)

- Handle duplicates

### Handle errors in variables

- Missing Data: can happen due to forgotten to store, inappropriate data handling, inefficient data entry at the ground level, etc. 

- High Cardinality: the number of different labels in categorical data is very high, which causes problems to model to learn.

- Outliers: the extreme cases that may be due to error, but not in every case.


## Data Cleaning

Data cleaning refers to identifying and correcting errors in the dataset that may negatively impact a predictive model.

- Identify Columns That Contain a Single Value
- Delete Columns That Contain a Single Value
- Consider Columns That Have Very Few Values
- Remove Columns That Have A Low Variance
- Identify Rows that Contain Duplicate Data
- Delete Rows that Contain Duplicate Data

Data cleaning also includes the following [2]:

1. Handle missing values
2. Handle Outliers
3. Handle categorical data
3. Encoding class labels
4. Scaling and normalization
5. Parsing dates
6. Character encodings
7. Inconsistent Data Entry

### Check data types

```py
  df.info()
```

```py
    # List of numeric columns
    num_cols = ['age', 'bp', 'sg', 'al', 'su',
                'bgr', 'bu', 'sc', 'sod', 'pot',
                'hemo', 'pcv', 'wbcc', 'rbcc']
                
    for column in df.columns:
        if column in num_cols:
            # Replace ‘?’ with ‘NaN’ 
            # df[column] = df[column].replace('?', np.nan)
            
            # Convert to numeric type
            df[column] = pd.to_numeric(df[column])
```

### Handle missing values

The removal of samples or dropping of feature columns may not feasible because we might lose too much valuable data. 

We can use interpolation techniques to estimate the missing values from the other training samples in the dataset.

One of the most common interpolation techniques is _mean imputation_ where we simply replace the missing value by the mean value of the entire feature column

- Numerical Imputation
- Categorical Imputation

Check for null values. 
We can drop or fill the `NaN` values.

```py
    # return the number of missing values (NaN) per column
    df.isnull().sum()  

    # remove all rows that contain a missing value
    df.dropna()
    
    # remove all columns with at least one missing value
    df.dropna(axis=1)
    
    # Drop the NaN
    df['col_name'] = df['col_name'].dropna(axis=0, how="any")

    # check NaN again
    df['col_name'].isnull().sum()
    
    # remove rows with None in column "date"
    # notna is much faster
    df.dropna(subset=['date'])
    df = df[df["date"].notna()]
```

```py
    # We can delete specific columns by passing a list
    df.dropna(subset=['City', 'Shape Reported'], how='all')

    # Replace NaN by a specific value using fillna() method
    df['Shape Reported'].isna().sum()

    df['Shape Reported'].fillna(value='VARIOUS', inplace=True)
    df['Shape Reported'].isna().sum()
    df['Shape Reported'].value_counts(dropna=False)
```

### Handle duplicate values

```py
    # We can show if there are duplicates in specific column 
    # by calling 'duplicated' on a Series.
    df.zip_code.duplicated().sum()

    # Check if an entire row is duplicated 
    df.duplicated().sum()

    # find duplicate rows across all columns
    dup_rows = df[df.duplicated()]
    
    # find duplicate rows across specific columns
    dup_rows = df[df.duplicated(['col1', 'col2'])]
     
    # Return DataFrame with duplicate 
    # rows removed, optionally only considering certain columns.
    # 'keep' controls the rows to keep.
    df.drop_duplicates(keep='first').shape
    
    # extract date column and remove None values
    # drop_duplicates is faster on larger dataframes
    date = df[df["date"].notna()]
    date_set = date.drop_duplicates(subset=['date'])['date'].values
    
    # extract date column and remove None values
    date = df[df["date"].notna()]['date'].values
    date_set = np.unique(date)
``` 


### Handle Outliers

- Remove: Outlier entries are deleted from the distribution

- Replace: The outliers could be handled as missing values and replaced with suitable imputation.

- Cap: Using an arbitrary value or a value from a variable distribution to replace the maximum and minimum values.

- Discretize: Converting continuous variables into discrete values. 


```py
    # making boolean series for a team name
    filter1 = data["Team"] == "Atlanta Hawks"

    # making boolean series for age
    filter2 = data["Age"] > 24

    # filtering data on basis of both filters
    df.where(filter1 & filter2, inplace=True)

    df.loc[filter1 & filter2]

    # display
    print(df.head(20))
```


## Encoding Categorical Features

Machine learning algorithms and deep learning neural networks require that input and output variables are numbers.

This means that categorical data must be encoded to numbers before we can use it to fit and evaluate a model.

For categorical data, we need to distinguish between _nominal_ and _ordinal_ features. 

Ordinal features can be understood as categorical values that can be sorted or ordered. For example, T-shirt size would be an ordinal feature because we can define an order XL > L > M. 

Nominal features do not imply any order. Thus, T-shirt color is a nominal feature since it typically does not make sense to say that red is larger than blue.

There are many ways to encode categorical variables:

  1. Integer (Ordinal) Encoding: each unique label/category is mapped to an integer.
  2. One Hot Encoding: each label is mapped to a binary vector.
  3. Dummy Variable Encoding
  4. Learned Embedding: a distributed representation of the categories is learned.

### Integer (Ordinal) Encoding

To make sure that the ML algorithm interprets the ordinal features correctly, we need to convert the categorical string values into integers. 

Unfortunately, there is no convenient function that can automatically derive the correct order of the labels of our size feature. 

Thus, we have to define the mapping manually.

```py
    size_mapping = { 'XL': 3, 'L': 2, 'M': 1}
    df['size'] = df['size'].map(size_mapping)
```

### Encoding Class Labels

Many machine learning libraries require that class labels are encoded as integer values. 

It is considered good practice to provide class labels as integer arrays to avoid technical glitches. 

```py
    # Handle categorical features
    df['is_white_wine'] = [1 if typ == 'white' else 0 for typ in df['type']]

    # Convert to a binary classification task
    df['is_good_wine'] = [1 if quality >= 6 else 0 for quality in df['quality']]

    df.drop(['type', 'quality'], axis=1, inplace=True)
```

To encode the class labels, we can use an approach similar to the mapping of ordinal features above. 

We need to remember that class labels are not ordinal so it does not matter which integer number we assign to a particular string-label.

There is a convenient `LabelEncoder` class in scikit-learn to achieve the same results as _map_.  

```py
    from sklearn.preprocessing import LabelEncoder
    class_le = LabelEncoder()
    y = class_le.fit_transform(df['classlabel'].values)
    # array([0, 1, 0])

    class_le.inverse_transform(y)
    # array(['class1', 'class2', 'class1'], dtype=object)
```

### One-Hot Encoding of Nominal Features

A one-hot encoding is a type of encoding in which an element of a finite set is represented by the index in that set where only one element has its index set to “1” and all other elements are assigned indices within the range [0, n-1]. 

In contrast to binary encoding schemes where each bit can represent 2 values (0 and 1), one-hot encoding assigns a unique value to each possible value.

In the previous section, we used a simple dictionary-mapping approach to convert the ordinal size feature into integers. 

Since scikit-learn's estimators treat class labels without any order, we can use the convenient `LabelEncoder` class to encode the string labels into integers.

```py
    X = df[['color', 'size', 'price']].values
    color_le = LabelEncoder()
    X[:, 0] = color_le.fit_transform(X[:, 0])
```

After executing the code above, the first column of the NumPy array X now holds the new color values which are encoded as follows: blue = 0, green = 1, red = 2 n

However, we will make one of the most common mistakes in dealing with categorical data. Although the color values are not ordered, a ML algorithm will now assume that green is larger than blue, and red is larger than green. Thus, the results would not be optimal.

A common workaround is to use a technique called _one-hot encoding_ to create a new dummy feature for each unique value in the nominal feature column. 

Here, we would convert the color feature into three new features: blue, green, and red. 

Binary values can then be used to indicate the particular color of a sample; for example, a blue sample can be encodedas blue=1, green=0, red=0. 

We can use the `OneHotEncoder` that is implemented in the scikit-learn.preprocessing module. 

An even more convenient way to create those dummy features via one-hot encoding is to use the get_dummies method implemented in pandas. Applied on a DataFrame, the get_dummies method will only convert string columns and leave all other columns unchanged:

```py
    pd.get_dummies(df[['price', 'color', 'size']])
```

### Dummy Variable Encoding

Most machine learning algorithms cannot directly handle categorical features that are _text values_.

Therefore, we need to create dummy variables for our categorical features which is called _one-hot encoding_.

The one-hot encoding creates one binary variable for each category which includes redundancy. 

In contrast, a dummy variable encoding represents N categories with N-1 binary variables.

```py
    pd.get_dummies(df, columns=['Color'], prefix=['Color'])
```

In addition to being slightly less redundant, a dummy variable representation is required for some models such as linear regression model (and other regression models that have a bias term) since a one hot encoding will cause the matrix of input data to become singular which means it cannot be inverted, so the linear regression coefficients cannot be calculated using linear algebra. Therefore, a dummy variable encoding must be used.

However, we rarely encounter this problem in practice when evaluating machine learning algorithms other than linear regression.


### Complete One-Hot Encoding Example

A one-hot encoding is appropriate for categorical data where no relationship exists between categories.

The scikit-learn library provides the `OneHotEncoder` class to automatically one hot encode one or more variables.

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


## Scaling vs Normalization

See **Normalization Techniques** in Feature Engineering.

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

### Normalization vs Standardization

_Feature scaling_ is a crucial step in a preprocessing pipeline that can easily be forgotten.

Decision trees and random forests are one of the few machine learning algorithms where we do not need to worry about feature scaling.

The majority of machine learning and optimization algorithms behave much better if features are on the same scale. 

- **Normalization:** All values are scaled in a specified range between 0 and 1 via normalization or min-max scaling. 

- **Standardization:** The process of scaling values while accounting for standard deviation or z-score normalization.

  If the standard deviation of features differs, the range of those features will also differ. Therefore, the effect of outliers is reduced. 

  To arrive at a distribution with a 0 mean and 1 variance, all the data points are subtracted by their mean and the result divided by the distribution’s variance.

> Note that we fit the `StandardScaler` on the training data then use those parameters to transform the test set or any new data point.

> Regularization is another reason to use feature scaling such as standardization. For regularization to work properly, all features must be on comparable scales.


### How to Choose between Standardization vs Normalization

Data-centric heuristics include the following:

1. If your data has outliers, use standardization or robust scaling.
2. If your data has a gaussian distribution, use standardization.
3. If your data has a non-normal distribution, use normalization.

Model-centric rules include these:

1. If your modeling algorithm assumes (but does not require) a normal distribution of the residuals (such as regularized linear regression, regularized logistic regression, or linear discriminant analysis), use standardization.

2. If your modeling algorithm makes no assumptions about the distribution of the data (such as k-nearest neighbors, support vector machines, and artificial neural networks), then use normalization.

In each use case, the rule proposes a mathematical fit with either the data or the learning model. 


Normalization does not affect the feature distribution, but it does exacerbate the effects of outliers due to lower standard deviations. Thus, outliers should be dealt with prior to normalization.

Standardization can be more practical for many machine learning algorithms since many linear models such as logistic regression and SVM initialize the weights to 0 or small random values close to 0. Using standardization, we center the feature columns at mean 0 with standard deviation 1 so that the feature columns take the form of a normal distribution which makes it easier to learn the weights.

In addition, standardization maintains useful information about outliers and makes the algorithm less sensitive to them whereas min-max only scales the data to a limited range of values.


### Log Transform

Log Transform is the most used technique among data scientists to turn a skewed distribution into a normal or less-skewed distribution. 

We take the log of the values in a column and utilize those values as the column in this transform. 

Log transform is used to handle confusing data so that the data becomes more approximative to normal applications.


## Normalization Techniques

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

We can also use `RobustScaler` when we want to reduce the effects of outliers compared to `MinMaxScaler`.


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

We can apply the standardization in Pandas using the `.mean()` and `.std()` methods which preserves the column headers/names.

```py
    # copy the data
    df_z_scaled = df.copy()
      
    # apply normalization from scratch
    for column in df_z_scaled.columns:
        df_z_scaled[column] = (df_z_scaled[column] - df_z_scaled[column].mean()) / df_z_scaled[column].std()    
```


## Parsing dates

Method 1: Parse date columns using `read_csv`

```py
    def parser(x):
        return dt.datetime.strptime(x, "%Y-%m-%d")

    def load_data(name):
        df_data = pd.read_csv(
            file_path,
            header=0,
            index_col=0,
            parse_dates=["day"],
            date_parser=parser    # optional
        )
        
        return df_data
```

Method 2: Parse dates using `to_datetime`

```py
    def load_data(name):
        df_data = pd.read_csv(name, header=3, index_col=0)

        # Replace index with DateTime
        df_data.index = pd.to_datetime(df_data.index)
        
        return df_data
```


## Inconsistent Data Entry

TODO: This will most likely vary 


## Highly Imbalanced Data

Need to upsample, but categories with only 1 entry when oversampled will give a 100% accuracy and artificially inflate the total accuracy/precision.

- We can use `UpSample` in Keras/PyTorch and `pd.resample()`


## Order of Data Transforms for Time Series

You may want to experiment with applying multiple data transforms to a time series prior to modeling.

It is quite common to;

- apply a power transform to remove an increasing variance

- apply seasonal differencing to remove seasonality

- apply one-step differencing to remove a trend.

The order that the transform operations are applied is important.


## Train-Test Split

Also see [Train-Test Split](./ml/train_teat_split.md)

A key step in ML is the choice of model.  

> Split first, normalize later.

A train-test split conists of the following:

1. Split the dataset into training, validation and test set

2. We normalize the training set only (fit_transform). 

3. We normalize the validation and test sets using the normalization factors from train set (transform).


> Instead of discarding the allocated test data after model training and evaluation, it is a good idea to retrain a classifier on the entire dataset for optimal performance.


----------


## Data Pipelines

There are multiple stages to running machine learning algorithms since it involves a sequence of tasks including pre-processing, feature extraction, model fitting, performance, and validation.

**Pipeline** is a technique used to create a linear sequence of data preparation and modeling steps to automate machine learning workflows [3].

Pipelines help in parallelization which means different jobs can be run in parallel as well as help to inspect and debug the data flow in the model.

Here we are using the Pipeline class from scikit-learn. 

### Import Libraries

```py
    import pandas as pd
    import numpy as np

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
    from sklearn.decomposition import PCA
```

```py
    # Load dataset
    data_df=pd.read_csv("Path to CSV file")
    
    # Drop null values
    data_df.dropna()
    
    data_df.head()
    
    # Calculate the value counts for each category
    data_df['R'].value_counts()
    
    # Find the unique values
    data_df['R'].unique()
    
    # Split the data into training and testing set
    x = data_df.drop(['R'], axis=1)
    y = data_df['R']
    trainX, testX, trainY, testY = train_test_split(x, y, test_size = 0.2)
```

### Create Simple Pipeline

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

Convert pipeline to DataFrame and show the best model:

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

### Pipeline with PCA

Pipeline example with Principal Component Analysis (PCA)


## More on scikit-learn pipeline

A scikit-learn pipeline is a component provided by scikit-learn package that allow us to merge different components within the scikit-learn API and run them sequentially. 

Thus, pipelines are very helpful because the component can perform all the data preprocessing and model fitting and also help to minimize human error during the data transformation and fitting process [5][6]. 

### Data Prep

```py
    # convert question mark '?' to NaN
    df.replace('?', np.nan, inplace=True)
        
    # convert target column from string to number
    le = LabelEncoder()
    df.income = le.fit_transform(df.income)
```

### Create Pipeline

```py
    # create column transformer component
    # We will select and handle categorical and numerical features in a differently
    preprocessor = ColumnTransformer([
        ('numerical', numerical_pipe, make_column_selector(dtype_include=['int', 'float'])),
        ('categorical', categorical_pipe, make_column_selector(dtype_include=['object'])),
        ])

    # create pipeline for numerical features
    numerical_pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # create pipeline for categorical features
    categorical_pipe = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy='most_frequent')),
        ('one_hot', OneHotEncoder(handle_unknown='ignore'))
        ])

        
    # create main pipeline
    pipe = Pipeline([
        ('column_transformer', preprocessor),
        ('model', KNeighborsClassifier())
        ])
```

### Train and Evaluate Pipeline

```py
    # create X and y variables
    X = df.drop('income', axis=1)
    y = df.income

    # split data into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # fit pipeline with train data and predicting test data
    pipe.fit(X_train, y_train)
    predictions = pipe.predict(X_test)

    # check pipeline accuracy
    accuracy_score(y_test, predictions)
```

The main advantage of using this pipelines is that we can save them just like any other model in scikit-learn. 

The scikit-learn pipelines are estimators, so we can save them with all the preprocessing and modelling steps into a binary file using joblib and load them from the binary file later. 

```py
    import joblib

    # Save the pipeline into a binary file
    joblib.dump(pipe, 'wine_pipeline.bin')

    # Load the saved pipeline from a binary file
    pipe = joblib.load('wine_pipeline.bin')
```


### Joblib

**Joblib** is an open-source Python library that helps to save pipelines to a file that can be used later [7].

### Pandas pipe

The pandas `pipe` function offers a structured and organized way for combining several functions into a single operation.

As the number of steps increase, the syntax becomes cleaner with the pipe function compared to executing functions 



## Bootstrapping

The goal of bootstrap is to create an estimate (sample mean x̄) for a population parameter (population mean θ) based on multiple data samples obtained from the original sample.

Bootstrapping is done by repeatedly sampling (with replacement) the sample dataset to create many simulated samples. 

Each simulated bootstrap sample is used to calculate an estimate of the parameter and the estimates are then combined to form a sampling distribution.

The bootstrap sampling distribution then allows us to draw statistical inferences such as estimating the standard error of the parameter.


----------



## Code Examples and References

### Data Preparation

[Data Science Primer](https://elitedatascience.com/primer)

[Tour of Data Preparation Techniques for Machine Learning](https://machinelearningmastery.com/data-preparation-techniques-for-machine-learning/)


[How to Perform Data Cleaning for Machine Learning with Python?](https://machinelearningmastery.com/basic-data-cleaning-for-machine-learning/)

[Preprocessing of the data using Pandas and SciKit](https://mclguide.readthedocs.io/en/latest/sklearn/preprocessing.html)

[Missing Values Be Gone](https://towardsdatascience.com/missing-values-be-gone-a135c31f87c1?source=rss----7f60cf5620c9---4&gi=d11a8ff041dd)

[ML Guide Quick Reference](https://mclguide.readthedocs.io/en/latest/sklearn/guide.html)

[The Lazy Data Scientist’s Guide to AI/ML Troubleshooting](https://medium.com/@ODSC/the-lazy-data-scientists-guide-to-ai-ml-troubleshooting-abaf20479317?source=linkShare-d5796c2c39d5-1638394993&_branch_referrer=H4sIAAAAAAAAA8soKSkottLXz8nMy9bLTU3JLM3VS87P1Xcxy8xID4gMc8lJAgCSs4wwIwAAAA%3D%3D&_branch_match_id=994707642716437243)


[How to Select a Data Splitting Method](https://towardsdatascience.com/how-to-select-a-data-splitting-method-4cf6bc6991da)


### Categorical Data

[4 Categorical Encoding Concepts to Know for Data Scientists](https://towardsdatascience.com/4-categorical-encoding-concepts-to-know-for-data-scientists-e144851c6383)

[Ordinal and One-Hot Encodings for Categorical Data](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/)

[Smarter Ways to Encode Categorical Data for Machine Learning](https://towardsdatascience.com/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159)

[3 Ways to Encode Categorical Variables for Deep Learning](https://machinelearningmastery.com/how-to-prepare-categorical-data-for-deep-learning-in-python/)

[Stop One-Hot Encoding Your Categorical Variables](https://towardsdatascience.com/stop-one-hot-encoding-your-categorical-variables-bbb0fba89809)


### Scaling

[How to Selectively Scale Numerical Input Variables for Machine Learning](https://machinelearningmastery.com/selectively-scale-numerical-input-variables-for-machine-learning/)

[How to use Data Scaling Improve Deep Learning Model Stability and Performance](https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/)

[How to Transform Target Variables for Regression in Python](https://machinelearningmastery.com/how-to-transform-target-variables-for-regression-with-scikit-learn/)

[The Mystery of Feature Scaling is Finally Solved](https://towardsdatascience.com/the-mystery-of-feature-scaling-is-finally-solved-29a7bb58efc2?source=rss----7f60cf5620c9---4)


### Normalization

[How to Use StandardScaler and MinMaxScaler Transforms in Python](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/)

[How to Use Power Transforms for Machine Learning](https://machinelearningmastery.com/power-transforms-with-scikit-learn/)


### Train-Test Split

[Training-validation-test split and cross-validation done right](https://machinelearningmastery.com/training-validation-test-split-and-cross-validation-done-right/)

[A Gentle Introduction to k-fold Cross-Validation](https://machinelearningmastery.com/k-fold-cross-validation/)

[How to Configure k-Fold Cross-Validation](https://machinelearningmastery.com/how-to-configure-k-fold-cross-validation/)



## References

W. McKinney, Python for Data Analysis 2nd ed., Oreilly, ISBN: 978-1-491-95766-0, 2018.

[1] [INFOGRAPHIC: Data prep and Labeling](https://www.cognilytica.com/2019/04/19/infographic-data-prep-and-labeling/)

[2] [Kaggle Data Cleaning Challenge: Missing values](https://www.kaggle.com/rtatman/data-cleaning-challenge-handling-missing-values)


[3] [Build Machine Learning Pipelines](https://medium.datadriveninvestor.com/build-machine-learning-pipelines-with-code-part-1-bd3ed7152124?gi=c419327a3c8c)

[4] [A Better Way for Data Preprocessing: Pandas Pipe](https://towardsdatascience.com/a-better-way-for-data-preprocessing-pandas-pipe-a08336a012bc)

[5] [Introduction to Scikit-learn’s Pipelines](https://towardsdatascience.com/introduction-to-scikit-learns-pipelines-565cc549754a)

[6] [Unleash the Power of Scikit-learn’s Pipelines](https://towardsdatascience.com/unleash-the-power-of-scikit-learns-pipelines-b5f03f9196de)

[7] [Lightweight Pipelining In Python using Joblib](https://towardsdatascience.com/lightweight-pipelining-in-python-1c7a874794f4)


[Customizing Sklearn Pipelines: TransformerMixin](https://towardsdatascience.com/customizing-sklearn-pipelines-transformermixin-a54341d8d624?source=rss----7f60cf5620c9---4)
