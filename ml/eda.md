# Exploratory Data Analysis

[11 Essential Code Blocks for Complete EDA](https://towardsdatascience.com/11-simple-code-blocks-for-complete-exploratory-data-analysis-eda-67c2817f56cd)

## Basic data set Exploration

1. Shape (dimensions) of the DataFrame

2. Data types of the various columns

What to look out for;

- Numeric features that should be categorical and vice versa.

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

- Strongly correlated features; either dark red (positive) or dark blue(negative).
- Target variable: if it has strong positive or negative relationships with other features.
