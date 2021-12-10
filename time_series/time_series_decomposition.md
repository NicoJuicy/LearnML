<!-- MarkdownTOC -->

- How to Decompose Time Series Data
    - Combining Time Series Components
    - Classical decomposition
        - Additive Model
        - Multiplicative Model
    - Decomposition as a Tool
    - Automatic Time Series Decomposition
    - Airline Passengers Dataset
- Time Series 101 Guide Python
    - STL Decomposition
    - Basic Time Series Forecasting Merhods
- How To Isolate Trend, Seasonality, and Noise From A Time Series
    - A step-by-step procedure for decomposing a time series into trend, seasonal, and noise components
    - Time series decomposition using statsmodels
- Time Series Data Visualization with Python
    - Time Series Visualization
    - References

<!-- /MarkdownTOC -->

# How to Decompose Time Series Data

Time series analysis provides a useful abstraction for selecting forecasting methods which is the _decomposition_ of a time series into systematic and unsystematic components.

- **Systematic:** Components of the time series that have consistency or recurrence which can be described and modeled.

- **Non-Systematic:** Components of the time series that cannot be directly modeled.

A given time series is thought to consist of three systematic components: level, trend, and seasonality plus one non-systematic component called _noise_.

- Level: The average value in the series.

- Trend: The increasing or decreasing value in the series.

- Seasonality: The repeating short-term cycle in the series.

- Noise: The random variation in the series.


- Cyclic Patterns: A cyclic pattern is a repetitive pattern of the data that does not occur in a fixed period of time. These is usually a repetitive patterns that continuously repeats more than a year or longer.

- Signal: Signal is the real pattern, the repeatable process/pattern in data.


Time series analysis is concerned with using methods such as decomposition of a time series into its systematic components in order to understand the underlying causes or the _why_ behind the time series dataset which is usually not helpful for prediction.

## Combining Time Series Components

A time series is thought to be an aggregate or combination of these four components.

All series have a level and noise. 

The trend and seasonality components are optional.

It is also helpful to think of the components as combining either additive or multiplicative.

## Classical decomposition

The **classical decomposition** method originated in the 1920s. 

It is a relatively simple procedure and forms the starting point for most other methods of time series decomposition. 

There are two forms of classical decomposition: an additive decomposition and a multiplicative decomposition. 

The classical method of time series decomposition forms the basis of many time series decomposition methods, so it is important to understand how it works. 

The first step in a classical decomposition is to use a **moving average** method to estimate the trend-cycle.

### Additive Model

An additive model suggests that the components are added together as follows:

```
    y(t) = Level + Trend + Seasonality + Noise
```

An additive model is linear where changes over time are consistently made by the same amount.

A linear trend is a straight line.

A linear seasonality has the same frequency (width of cycles) and amplitude (height of cycles).

### Multiplicative Model

A multiplicative model suggests that the components are multiplied as follows:

```
    y(t) = Level * Trend * Seasonality * Noise
```

A multiplicative model is nonlinear () quadratic or exponential) in which changes increase or decrease over time.

A nonlinear trend is a curved line.

A nonlinear seasonality has an increasing or decreasing frequency and/or amplitude over time.


## Decomposition as a Tool

Decomposition is primarily used for time series analysis which can be used to inform forecasting models on your problem.

It provides a structured way of thinking about a time series forecasting problem, both generally in terms of modeling complexity and specifically in terms of how to best capture each of these components in a given model.

Each of these components are something you may need to think about and address during data preparation, model selection, and model tuning. 

You may address it explicitly in terms of modeling the trend and subtracting it from your data or implicitly by providing enough history for an algorithm to model a trend if it exists.

You may or may not be able to cleanly or perfectly break down your time series as an additive or multiplicative model.

- Real-world problems are messy and noisy.

- There may be additive and multiplicative components. 

- There may be an increasing trend followed by a decreasing trend. 

- There may be non-repeating cycles mixed in with the repeating seasonality components.

However, these abstract models provide a simple framework that you can use to analyze your data and explore ways to think about and forecast your problem.


## Automatic Time Series Decomposition

There are methods to automatically decompose a time series.

The `statsmodels` library provides an implementation of the naive or classical decomposition method in a function called `seasonal_decompose()`. 

The statsmodels linrary requires that you specify whether the model is additive or multiplicative.

Both techniques will produce a result and you must be careful to be critical when interpreting the result. 

A review of a plot of the time series and some summary statistics can often be a good start to get an idea of whether your time series problem looks additive or multiplicative.

The `seasonal_decompose()` function returns a result object which contains arrays to access four pieces of data from the decomposition.

The snippet below shows how to decompose a series into trend, seasonal, and residual components assuming an additive model.

```py
    from statsmodels.tsa.seasonal import seasonal_decompose
    series = ...
    result = seasonal_decompose(series, model='additive')
    print(result.trend)
    print(result.seasonal)
    print(result.resid)
    print(result.observed)
```

The result object provides access to the trend and seasonal series as arrays. It also provides access to the _residuals_ which are the time series after the trend and  seasonal components are removed. Finally, the original or observed data is also stored.

These four time series can be plotted directly from the result object by calling the `plot()` function.

```python
    from statsmodels.tsa.seasonal import seasonal_decompose
    from matplotlib import pyplot
    series = ...
    result = seasonal_decompose(series, model='additive')
    result.plot()
    pyplot.show()
```

Although classical methods are common, they are not recommended for the following reasons:

- The technique is not robust to outlier values.

- It tends to over-smooth sudden rises and dips in the time series data.

- It assumes that the seasonal component repeats from year to year.

- The method produces no trend-cycle estimates for the first and last few observations.

- Other better methods that can be used for decomposition are X11 decomposition, SEAT decomposition, or STL decomposition. We will now see how to generate them in Python.

STL has many advantages over classical, X11, and SEAT decomposition techniques. 


## Airline Passengers Dataset

The Airline Passengers dataset describes the total number of airline passengers over a period of time.

The units are a count of the number of airline passengers in thousands. There are 144 monthly observations from 1949 to 1960.

First, let us graph the raw observations.

Reviewing the line plot, it suggests that there may be a linear trend, but it is hard to be sure from eye-balling. 

There is also seasonality, but the amplitude (height) of the cycles appears to be increasing, suggesting that it is multiplicative.


We will assume a multiplicative model.

The example below decomposes the airline passenger dataset as a multiplicative model.

Running the example plots the observed, trend, seasonal, and residual time series.

We can see that the trend and seasonality information extracted from the series does seem reasonable. The residuals are also interesting, showing periods of high variability in the early and later years of the series.


----------


# Time Series 101 Guide Python

## STL Decomposition

STL stands for Seasonal and Trend decomposition using Loess. 

The method is robust to outliers and can handle any kind of seasonality which also makes it a versatile method for decomposition.

There are a few things you can control when using STL:

- Trend cycle smoothness

- Rate of changes in seasonal component

- The robustness towards the user outlier or exceptional values which will allow you to control the effects of outliers on the seasonal and trend components.

SLT has its disadvantages. 

- STL cannot handle calendar variations automatically. 

- STL only provides a decomposition for additive models. 

  You can get the multiplicative decomposition by first taking the logs of the data and then back transforming the components.

```py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
 
elecequip = read_csv(r"C:/Users/datas/python/data/elecequip.csv")
stl = STL(elecequip, period=12, robust=True)
res_robust = stl.fit()
fig = res_robust.plot()
```


## Basic Time Series Forecasting Merhods

Although there are many statistical techniques available for forecasting a time series data, we will only talk about the most straightforward and simple methods that can be used for effective time series forecasting. 

These methods will also serve as the foundation for some of the other methods.

- Simple Moving Average (SMA)
- Weighted Moving Average (WMA)
- Exponential Moving Average (EMA)


----------



# How To Isolate Trend, Seasonality, and Noise From A Time Series

The commonly occurring seasonal periods are a day, week, month, quarter (or season), and year.

## A step-by-step procedure for decomposing a time series into trend, seasonal, and noise components

There are many decomposition methods available ranging from simple moving average based methods to powerful ones such as STL.

In Python, the `statsmodels` library has a `seasonal_decompose()` method that lets you decompose a time series into trend, seasonality and noise in one line of code.

First, let us understand how decomposition  works. 

We can create the decomposition of a time series into its trend, seasonal and noise components using a simple procedure based on moving averages using the following steps:

STEP 1: Identify the length of the seasonal period
STEP 2: Isolate the trend
STEP 3: Isolate the seasonality+noise
STEP 4: Isolate the seasonality
STEP 5: Isolate the noise

We will use the following time series of retail sales of user cars dealers in the US:

## Time series decomposition using statsmodels

Now that we know how decomposition works from the inside, we can cheat a little and use `seasonal_decompose()` in statsmodels to do all of the above work in one line of code. 

```py
from statsmodels.tsa.seasonal import seasonal_decompose
 
components = seasonal_decompose(df['Retail_Sales'], model='multiplicative')

# components = seasonal_decompose(np.array(elecequip), model='multiplicative', freq=4)
 
components.plot()
```


----------


# Time Series Data Visualization with Python

## Time Series Visualization

Visualization plays an important role in time series analysis and forecasting.

Plots of the raw sample data can provide valuable diagnostics to identify temporal structures like trends, cycles, and seasonality that can influence the choice of model.

A problem is that many novices in the field of time series forecasting stop with line plots.

In this tutorial, we will take a look at 6 different types of visualizations that you can use on your own time series data. They are:

1. Line Plots
2. Histograms and Density Plots
3. Box and Whisker Plots
4. Heat Maps
5. Lag Plots or Scatter Plots
6. Autocorrelation Plots

The focus is on univariate time series, but the techniques are just as applicable to multivariate time series when you have more than one observation at each time step.


## Avoid Common Mistakes

[Avoid These Mistakes with Time Series Forecasting](https://www.kdnuggets.com/2021/12/avoid-mistakes-time-series-forecasting.html)

- How to find peaks and troughs in a time series signal?

- What is (and how to use) autocorrelation plot?

- How to check if a time series has any statistically significant signal?



## References

[How to Decompose Time Series Data into Trend and Seasonality](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/)

[Time Series 101 Guide Python](https://datasciencebeginners.com/2020/11/25/time-series-forecast-and-decomposition-101-guide-python/)

[How To Isolate Trend, Seasonality, and Noise From A Time Series](https://timeseriesreasoning.com/contents/time-series-decomposition/)

[Time Series Data Visualization with Python](https://machinelearningmastery.com/time-series-data-visualization-with-python/)

[Avoid These Mistakes with Time Series Forecasting](https://www.kdnuggets.com/2021/12/avoid-mistakes-time-series-forecasting.html)

