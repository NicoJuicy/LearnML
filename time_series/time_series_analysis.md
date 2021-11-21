# Time Series Analysis

## How to import time series in python?

```py
  # Import as Dataframe
  df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'])
```

You can also import it as a pandas Series with the date as index by specifying the `index_col` argument. 

## What is panel data?

Panel data is also a time-based dataset.

In addition to time series, it also contains one or more related variables that are measured for the same time periods.

Typically, the columns present in panel data contain explanatory variables that can be helpful in predicting the Y, provided those columns will be available at the future forecasting period.

## Visualizing a time series

We can use matplotlib to visualise the time series.

```py
# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

    plot_df(df, x=df.index, y=df.value, title='Monthly anti-diabetic drug sales in Australia from 1992 to 2008.')
```

##  Patterns in a time series

Any time series may be split into the following components: Base Level + Trend + Seasonality + Error

A trend is observed when there is an increasing or decreasing slope observed in the time series. 

Seasonality is observed when there is a distinct repeated pattern observed between regular intervals due to seasonal factors. 

The seasonality could be due to the month of the year, the day of the month, weekdays, or even time of day.

Another aspect to consider is _cyclic_ behaviour which happens when the rise and fall pattern in the series does not occur in fixed calendar-based intervals.


## Additive and multiplicative time series

Depending on the nature of the trend and seasonality, a time series can be modeled as an additive or multiplicative where each observation in the series can be expressed as either a sum or a product of the components:

Additive time series:       Value = Base Level + Trend + Seasonality + Error

Multiplicative Time Series: Value = Base Level x Trend x Seasonality x Error


## How to decompose a time series into its components?

We can perform a classical decomposition of a time series by considering the series as an additive or multiplicative combination of the base level, trend, seasonal index and the residual using the `seasonal_decompose` function in statsmodels.


## Stationary and Non-Stationary Time Series

Stationarity is a property of a time series. A stationary series is one where the values of the series is not a function of time.

The statistical properties of the series like mean, variance, and autocorrelation are constant over time. 

Autocorrelation of the series is nothing but the correlation of the series with its previous values, more on this coming up.

A stationary time series is devoid of seasonal effects as well.

Most statistical forecasting methods are designed to work on a stationary time series. 

The first step in the forecasting process is typically to do some transformation to convert a non-stationary series to stationary.


### How to make a time series stationary?

You can make series stationary by:

1. Differencing the series (once or more)
2. Take the log of the series
3. Take the nth root of the series
4. Combination of the above

The most common and convenient method to stationarize the series is by differencing the series at least once until it becomes approximately stationary.

In short, differencing the series is simply subtracting the next value by the current value: y = y(t) - y(t-1)

### Why make a non-stationary series stationary before forecasting?

Forecasting a stationary series is relatively easy and the forecasts are more reliable.

An important reason is that autoregressive forecasting models are essentially linear regression models that utilize the lag(s) of the series itself as predictors.

We know that linear regression works best if the predictors (X) are not correlated against each other. 

Thus, stationarizing the series solves this problem since it removes any persistent autocorrelation, thereby making the predictors(lags of the series) in the forecasting models nearly independent.

### How to test for stationarity?

The stationarity of a series can be established by looking at the plot of the series like we did earlier.

Another method is to split the series into two or more contiguous parts and computing the summary statistics like the mean, variance and the autocorrelation. 

If the stats are quite different, then the series is not likely to be stationary.

We nwed a method to quantitatively determine if a given series is stationary or not which can be done using statistical tests called "Unit Root Tests". 

There are multiple variations of unit root tests where the tests check if a time series is non-stationary and possess a unit root.

There are multiple implementations of Unit Root tests like:

- Augmented Dickey Fuller test (ADH Test)

- Kwiatkowski-Phillips-Schmidt-Shin – KPSS test (trend stationary)

- Philips Perron test (PP Test)

The most commonly used is the ADF test where the null hypothesis is the time series possesses a unit root and is non-stationary. 

Thus, if the P-Value in ADH test is less than the significance level (0.05), you reject the null hypothesis.


The KPSS test is used to test for trend stationarity. The null hypothesis and the P-Value interpretation are the opposite of ADH test.


### What is the difference between white noise and a stationary series?

Like a stationary series, the white noise is also not a function of time which means its mean and variance do not change over time. 

The difference is that the white noise is completely random with a mean of 0.

In white noise there is no pattern. 

Mathematically, a sequence of completely random numbers with mean zero is considered white noise.


## How to detrend a time series?

Detrending a time series is to remove the trend component from a time series. 

There are multiple approaches to extract the trend:

1. Subtract the line of best fit from the time series. 

  The line of best fit may be obtained from a linear regression model with the time steps as the predictor. For more complex trends, you may want to use quadratic terms (x^2) in the model.
  
2. Subtract the trend component obtained from time series decomposition we saw earlier.

3. Subtract the mean

4. Apply a filter like Baxter-King filter or the Hodrick-Prescott Filter to remove the moving average trend lines or the cyclical components.

## How to deseasonalize a time series?

There are multiple approaches to deseasonalize a time series:

1. Take a moving average with length as the seasonal window which will smoothen the series in the process.

2. Seasonal difference the series (subtract the value of previous season from the current value)

3. Divide the series by the seasonal index obtained from STL decomposition

If dividing by the seasonal index does not work well, try taking a log of the series and then do the deseasonalizing. 

You can later restore to the original scale by taking an exponential.


## How to test for seasonality of a time series?

The common method to test for seasonality is to plot the series and check for repeatable patterns in a fixed time interval (hourly, daily, weekly, monthly, yearly)

If you want a more definitive inspection of the seasonality, use the Autocorrelation Function (ACF) plot. 


## How to treat missing values in a time series?

Sometimes your time series will have missing dates/times which means the data was not captured or was not available for those periods. 

It could be that the measurement was zero on those days in which case you may fill up those periods with zero.

When it comes to time series, you should typically NOT replace missing values with the mean of the series, especially if the series is not stationary. 

What you could do instead for a quick and dirty workaround is to _forward-fill_ the previous value.

However, depending on the nature of the series, you want to try out multiple approaches before concluding. 

Some effective alternatives to imputation are:

- Backward Fill
- Linear Interpolation
- Quadratic interpolation
- Mean of nearest neighbors
- Mean of seasonal couterparts

To measure the imputation performance, we can manually introduce missing values to the time series, impute it with above approaches, and then measure the mean squared error of the imputed against the actual values.


## What is autocorrelation and partial autocorrelation functions?

Autocorrelation is simply the correlation of a series with its own lags. 

If a series is significantly autocorrelated, the previous values of the series (lags) may be helpful in predicting the current value.

Partial Autocorrelation also conveys similar information but it conveys the pure correlation of a series and its lag, excluding the correlation contributions from the intermediate lags.

### How to compute partial autocorrelation function?

The partial autocorrelation of lag (k) of a series is the coefficient of that lag in the autoregression equation of Y. 

The autoregressive equation of Y is the linear regression of Y with its own lags as predictors.

Example: if Y_t is the current series and Y_t-1 is the lag 1 of Y, the partial autocorrelation of lag 3 (Y_t-3) is the coefficient alpha_3 of Y_t-3 in the following equation:

## Lag Plots

A Lag plot is a scatter plot of a time series against a lag of itself which is normally used to check for autocorrelation. 

If there is any pattern existing in the series such as the one you see below then the series is autocorrelated. 

If there is no such pattern, the series is likely to be random white noise.


## How to estimate the forecastability of a time series?

The more regular and repeatable patterns a time series has, the easier it is to forecast. 

The **approximate entropy** can be used to quantify the regularity and unpredictability of fluctuations in a time series.

The higher the approximate entropy, the more difficult it is to forecast it.

**Sample Entropy** is similar to approximate entropy but is more consistent in estimating the complexity even for smaller time series. 

For example, a random time series with fewer data points can have a lower approximate entropy than a more regular time series whereas a longer random time series will have a higher approximate entropy.

Thus, sample entropy is a better alternative to approximate entropy


## Why and How to smoothen a time series?

Smoothening of a time series may be useful:

- To reduce the effect of noise in a signal get a fair approximation of the noise-filtered series.

- The smoothed version of series can be used as a feature to explain the original series itself.
- Visualize the underlying trend better

Ways to smoothen a series:

  1. Take a moving average

  2. Do a LOESS smoothing (Localized Regression)

  3. Do a LOWESS smoothing (Locally Weighted Regression)

  
## References

[A Comprehensive Guide to Time Series Analysis in Python](https://www.machinelearningplus.com/time-series/time-series-analysis-python/)

[5 Must-Know Terms in Time Series Analysis](https://towardsdatascience.com/5-must-know-terms-in-time-series-analysis-bf2455b2a87)



