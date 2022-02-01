# Memory Usage Tips

<!-- MarkdownTOC -->

- Effective use of Data Types
    - Numerical Features
    - DateTime
    - Categorical
        - Better performance with categoricals
        - Categorical Methods
    - Creating dummy variables for modeling
    - Converting Between String and Datetime
    - Typecasting while Reading Data
- How to Profile Memory Usage?
- Optimize Your Python Code
- Make It Easier to Work with Large Datasets
    - Read using Pandas in Chunks
    - Dask
    - Vaex
    - Modin
- References

<!-- /MarkdownTOC -->

Here are some resources to evaluate Python memory usage.

## Effective use of Data Types

Make effective use of data types to prevent crashing of memory.

When the size of the dataset is comparatively larger than memory using such libraries is preferred, but when dataset size comparatively equal or smaller to memory size, we can optimize the memory usage while reading the dataset. 

Here, we discuss how to optimize memory usage while loading the dataset using `read_csv()` or `read_excel()`.

Using  `df.info()` we can view the default data types and memory usage.

### Numerical Features

For all numerical values, Pandas assigns float64 data type to a feature column having at least one float value, and int64 data type to a feature column having all feature values as integers.

Here is a list of the ranges of each datatype:

NOTE: A value with data type as int8 takes 8x times less memory compared to int64 data type.

### DateTime

By default, datetime columns are assigned as object data types that can be downgraded to DateTime format.

### Categorical

Pandas assign non-numerical feature columns as object data types which can be downgraded to category data types.

The non-numerical feature column usually has categorical variables which are mostly repeating. 

For example, the gender feature column has just 2 categories ‘Male’ and ‘Female’ that are repeating over and over again for all the instances which are re-occupying the space. 

Assigning gender to category datatype is a more compact representation.

**Better performance with categoricals**

If you do a lot of analytics on a particular dataset, converting to categorical can yield substantial overall performance gains. 

A categorical version of a DataFrame column will often use significantly less memory.

#### Better performance with categoricals

If you do a lot of analytics on a particular dataset, converting to categorical can yield substantial overall performance gains. A categorical version of a DataFrame column will often use significantly less memory, too.

#### Categorical Methods

Series containing categorical data have several special methods similar to the `Series.str` specialized string methods. This also provides convenient access to the categories and codes. 

The special attribute cat provides access to categorical methods:

```py
  s = pd.Series(['a', 'b', 'c', 'd'] * 2)
  
  cat_s = s.astype('category')
  cat_s.cat.codes
  cat_s.cat.categories
  
  
  actual_categories = ['a', 'b', 'c', 'd', 'e']

  cat_s2 = cat_s.cat.set_categories(actual_categories)
  cat_s2.value_counts()
```

In large datasets, categoricals are often used as a convenient tool for memory savings and better performance. After you filter a large DataFrame or Series, many of the categories may not appear in the data.

```py
  cat_s3 = cat_s[cat_s.isin(['a', 'b'])]
  cat_s3.cat.remove_unused_categories()
```

Table 12-1: Categorical methods for Series in pandas

### Creating dummy variables for modeling

When using statistics or machine learning tools, we usually transform categorical data into dummy variables callwd _one-hot encoding_ which involves creating a DataFrame with a column for each distinct category; these columns contain 1s for occurrences of a given category and 0 otherwise.

```py
  cat_s = pd.Series(['a', 'b', 'c', 'd'] * 2, dtype='category')

  pd.get_dummies(cat_s)
```

11. Time Series

11.1 Date and Time Data Types and Tools

11.2 Time Series Basics

11.3 Date Ranges, Frequencies, and Shifting

11.5 Periods and Period Arithmetic

11.6 Resampling and Frequency Conversion

11.7 Moving Window Functions

Table 11-1: Types in datetime module

Table 11-2: Datetime format specification (ISO C89 compatible)

Table 11-3: Locale-specific date formatting

Table 11-4: Base time series frequencies (not comprehensive)

Table 11-5. Resample method arguments


### Converting Between String and Datetime

You can format datetime objects and pandas Timestamp objects as strings using `str` or the `strftime` method passing a format specification. 

```py
  stamp = datetime(2011, 1, 3)

  str(stamp)
  stamp.strftime('%Y-%m-%d')
```

You can use many of the same format codes to convert strings to dates using `datetime.strptime`. 

```py
  value = '2011-01-03'

  datetime.strptime(value, '%Y-%m-%d')

  datestrs = ['7/6/2011', '8/6/2011']

  [datetime.strptime(x, '%m/%d/%Y') for x in datestrs]
```

pandas is generally oriented toward working with arrays of dates whether used as an axis index or a column in a DataFrame. 

The `to_datetime` method parses many different kinds of date representations. It also handles values that should be considered missing (None, empty string, etc.). 

NaT (Not a Time) is pandas’s null value for timestamp data.


12.2 Advanced GroupBy Use

12.3 Techniques for Method Chaining

The pipe Method

You can accomplish a lot with built-in pandas functions and the approaches to method chaining with callables that we just looked at. 

Sometimes you need to use your own functions or functions from third-party libraries. 

```py
  a = f(df, arg1=v1)
  b = g(a, v2, arg3=v3)
  c = h(b, arg4=v4)

  result = (df.pipe(f, arg1=v1)
            .pipe(g, v2, arg3=v3)
            .pipe(h, arg4=v4))
```

The statement `f(df)` and `df.pipe(f)` are equivalent but `pipe` makes chained invocation easier.


### Typecasting while Reading Data

The `read_csv` function includes a type parameter which accepts user-provided data types in a key-value format that can use instead of the default ones. 

The DateTime feature column can be passed to the `parse_dates` parameter.

```py
    dtype_dict = {
        'vendor_id': 'int8',
        'passenger_count': 'int8',
        'pickup_longitude': 'float16',
        'pickup_latitude': 'float16',
        'dropoff_longitude': 'float16',
        'dropoff_latitude': 'float16',
        'store-and_fwd_flag': 'category',
        'trip_duration': 'int32'
    }

    dates = ['pickup_datetime', 'dropoff_datetime']

    df = pd.read_csv("../data/train.csv",
                     dtype=dtype_dict,
                     parse_dates=dates)

    print(df.shape)
    print(df.info(verbose=False, memory_usage='deep'))
```


## How to Profile Memory Usage?

[How Much Memory is your ML Code Consuming?](https://towardsdatascience.com/how-much-memory-is-your-ml-code-consuming-98df64074c8f)

Learn how to quickly check the memory footprint of your machine learning function/module with one line of command. Generate a nice report too.

[Profile Memory Consumption of Python functions in a single line of code](https://towardsdatascience.com/profile-memory-consumption-of-python-functions-in-a-single-line-of-code-6403101db419)

Monitor line-by-line memory usage of functions with memory profiler module



## Optimize Your Python Code

1. Use the Built-in Functions rather than Coding them from Scratch

Some built-in functions in Python like map(), sum(), max(), etc. are implemented in C so they are not interpreted during the execution which saves a lot of time.

For example, if you want to convert a string into a list you can do that using the `map()` function instead of appending the contents of the strings into a list manually.

```py
    string = ‘Australia’
    U = map(str, s)
    print(list(string))
    # [‘A’, ‘u’, ‘s’, ‘t’, ‘r’, ‘a’, ‘l’, ‘i’, ‘a’]
```

Also, the use of f-strings while printing variables in a string instead of the traditional ‘+’ operator is also very useful in this case.


2. Focus on Memory Consumption During Code Execution

Reducing the memory footprint in your code definitely make your code more optimized. 

Check if unwanted memory consumption is occuring. 

Example: str concatenation using + operator will generate a new string each time which will cause unwanted memory consumption. Instead of using this method to concatenate strings, we can use the function `join()` after taking all the strings in a list.

3. Memoization in Python

Those who know the concept of dynamic programming are well versed with the concept of memorization. 

In memorization, the repetitive calculation is avoided by storing the values of the functions in the memory. 

Although more memory is used, the performance gain is significant. Python comes with a library called `functools` that has an LRU cache decorator that can give you access to a cache memory that can be used to store certain values.

4. Using C libraries/PyPy to Get Performance Gain

If there is a C library that can do your job then it’s better to use that to save time when the code is interpreted. 

The best way to do that is to use the ctype library in python. 

There is another library called CFFI which provides an elegant interface to C.

If you do not want to use C then you could use the PyPy package due to the presence of the JIT (Just In Time) compiler which gives a significant boost to your Python code.

5. Proper Use of Data Structures and Algorithms

This is more of a general tip but it is the most important one as it can give you a considerable amount of performance boost by improving the time complexity of the code.

For example, It is always a good idea to use dictionaries instead of lists in python in case you don’t have any repeated elements and you are going to access the elements multiple times.

This is because the dictionaries use hash tables to store the elements which have a time complexity of O(1) when it comes to searching compared to O(n) for lists in the worst case. So it will give you a considerable performance gain.


## Make It Easier to Work with Large Datasets

Pandas mainly uses a single core of CPU to process instructions and does not take advantage of scaling up the computation across various cores of the CPU to speed up the workflow. 

Thus, Pandas can cause memory issues when reading large datasets since it fails to load larger-than-memory data into RAM.

There are various other Python libraries that do not load the large data at once but interacts with system OS to map the data with Python. Further, they utilize all the cores of the CPU to speed up the computations. In this article, we will discuss 4 such Python libraries that can read and process large-sized datasets.

1. Pandas with chunks
2. Dask
3. Vaex
4. Modin

### Read using Pandas in Chunks

Pandas loads the entire dataset into RAM which may cause a memory overflow issue while reading large datasets.

Instead, we can read the large dataset in _chunks_ and perform data processing for each chunk.

The idea is to load 10k instances in each chunk (lines 11–14), perform text processing for each chunk (lines 15–16), and append the processed data to the existing CSV file (lines 18–21).

```py
# append to existing CSV file or save to new file
def saveDataFrame(data_temp):
    
    path = "DATA/text_dataset.csv"
    if os.path.isfile(path):
        with open(path, 'a') as f:
            data_temp.to_csv(f, header=False)
    else:
        data_temp.to_csv(path, index=False)
        
# Define chunksize
chunk_size = 10**3

# Read and process the dataset in chunks
for chunk in tqdm(pd.read_csv("DATA/text_dataset.csv", chunksize=chunk_size)):
    preprocessed_review = preprocess_text(chunk['review'].values)
     saveDataFrame(pd.DataFrame({'preprocessed_review':preprocessed_review, 
           'target':chunk['target'].values
         }))
```

### Dask

Dask is an open-source Python library that provides multi-core and distributed parallel execution of larger-than-memory datasets

Dask provides the high-performance implementation of the function that parallelizes the implementation across all the cores of the CPU.

Dask provides API similar to Pandas and Numpycwhich makes it easy for developers to switch between the libraries.

```py
import dask.dataframe as dd

# Read the data using dask
df_dask = dd.read_csv("DATA/text_dataset.csv")

# Parallelize the text processing with dask
df_dask['review'] = df_dask.review.map_partitions(preprocess_text)
```

### Vaex

Vaex is a Python library that uses an _expression system_ and _memory mapping_ to interact with the CPU and parallelize the computations across various cores of the CPU.

Instead of loading the entire data into memory, Vaex just memory maps the data and creates an expression system.

Vaex covers some of the API of pandas and is efficient to perform data exploration and visualization for a large dataset on a standard machine.

```py
import vaex

# Read the data using Vaex
df_vaex = vaex.read_csv("DATA/text_dataset.csv")

# Parallize the text processing
df_vaex['review'] = df_vaex.review.apply(preprocess_text)
```

### Modin

In contrast to Pandas, Modin utilizes all the cores available in the system, to speed up the Pandas workflow, only requiring users to change a single line of code in their notebooks.

```py
import modin.pandas as md

# read data using modin
modin_df = pd.read_csv("DATA/text_dataset.csv")

# Parallel text processing of review feature 
modin_df['review'] = modin_df.review.apply(preprocess_text)
```



## References

W. McKinney, Python for Data Analysis 2nd ed., Oreilly, ISBN: 978-1-491-95766-0, 2018. 

[Optimize Pandas Memory Usage for Large Datasets](https://towardsdatascience.com/optimize-pandas-memory-usage-while-reading-large-datasets-1b047c762c9b)

[Try These 5 Tips To Optimize Your Python Code](https://towardsdatascience.com/try-these-5-tips-to-optimize-your-python-code-c7e0ccdf486a?source=rss----7f60cf5620c9---4)

[4 Python Libraries that Make It Easier to Work with Large Datasets](https://towardsdatascience.com/4-python-libraries-that-ease-working-with-large-dataset-8e91632b8791)
