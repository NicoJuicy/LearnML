# Memory Usage Tips

Here are some resources to evaluate Python memory usage.


## Effective use of Data Types

[Optimize Pandas Memory Usage for Large Datasets](https://towardsdatascience.com/optimize-pandas-memory-usage-while-reading-large-datasets-1b047c762c9b)

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



## References

[Try These 5 Tips To Optimize Your Python Code](https://towardsdatascience.com/try-these-5-tips-to-optimize-your-python-code-c7e0ccdf486a?source=rss----7f60cf5620c9---4)


