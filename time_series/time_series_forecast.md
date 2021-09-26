# Time Series Forecasting

## Introduction

[What Is Time Series Forecasting?](https://machinelearningmastery.com/time-series-forecasting/)

[Taxonomy of Time Series Forecasting Problems](https://machinelearningmastery.com/taxonomy-of-time-series-forecasting-problems/)


## Time Series Analysis vs Time Series Forecasting

**Describing vs Predicting**

Time series analysis is concerned with using methods such as decomposition of a time series into its systematic components in order to understand the underlying causes or the _why_ behind the time series dataset which is usually not helpful for prediction.

Time series Forecasting is making predictionf about the future which is called _extrapolation_ in the classical statistical handling of time series data.

Forecasting involves taking models fit on historical data and using them to predict future observations.

Time series analysis can be used to remove trend and/or seasonality components which can help with forecasting.


## Time Series Decomposition

[How to Decompose Time Series Data into Trend and Seasonality](https://machinelearningmastery.com/decompose-time-series-data-trend-seasonality/)

[How To Isolate Trend, Seasonality, and Noise From A Time Series](https://timeseriesreasoning.com/contents/time-series-decomposition/)

[Time Series 101 Guide Python](https://datasciencebeginners.com/2020/11/25/time-series-forecast-and-decomposition-101-guide-python/)

[Time Series Data Visualization with Python](https://machinelearningmastery.com/time-series-data-visualization-with-python/)


## Data Preparation

[How to Load and Explore Time Series Data in Python](https://machinelearningmastery.com/load-explore-time-series-data-python/)

[Basic Feature Engineering With Time Series Data in Python](https://machinelearningmastery.com/basic-feature-engineering-time-series-data-python/)

[How To Backtest Machine Learning Models for Time Series Forecasting](https://machinelearningmastery.com/backtest-machine-learning-models-time-series-forecasting/)

NOTE: k-fold Cross Validation Does Not Work for Time Series Data.

The goal of time series forecasting is to make accurate predictions about the future.

The fast and powerful methods that we rely on in machine learning (such as using train-test splits and k-fold cross validation) do not work in the case of time series data since they ignore the temporal components inherent in the problem.


## Forecast Performance Baseline

[How to Make Baseline Predictions for Time Series Forecasting with Python](https://machinelearningmastery.com/persistence-time-series-forecasting-with-python/)

[How to Create an ARIMA Model for Time Series Forecasting in Python](https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/)

A _baseline_ in forecast performance provides a point of comparison.

- A baseline is a point of reference for all other modeling techniques on your problem. 

- If a model achieves performance at or below the baseline, the technique should be fixed or abandoned.

The technique used to generate a forecast to calculate the baseline performance must be easy to implement and naive of problem-specific details.

Before you can establish a performance baseline on your forecast problem, you must develop a test harness which is comprised of:

- The dataset you intend to use to train and evaluate models.

- The resampling technique you intend to use to estimate the performance of the technique (e.g. train/test split).

- The performance measure you intend to use to evaluate forecasts (e.g. mean squared error).

Then you need to select a naive technique that you can use to make a forecast and calculate the baseline performance.

The goal is to get a baseline performance on your time series forecast problem as quickly as possible so that you can get to work better understanding the dataset and developing more advanced models.

Three properties of a good technique for making a baseline forecast are:

- Simple: A method that requires little or no training or intelligence.

- Fast: A method that is fast to implement and computationally trivial to make a prediction.

- Repeatable: A method that is deterministic, meaning that it produces an expected output given the same input.

A common algorithm used in establishing a baseline performance is the _persistence algorithm_.


## 5-Step Forecasting Task

The 5 basic steps in a forecasting task are summarized by Hyndman and Athana­sopou­los in their book Forecasting: principles and practice. These steps are:

1. Problem Definition. The careful consideration of who requires the forecast and how the forecast will be used. 

This is described as the most difficult part of the process, most likely because it is entirely problem specific and subjective.

2. Gathering Information. The collection of historical data to analyze and model. 

This also includes getting access to domain experts and gathering information that can help to best interpret the historical information and ultimately the forecasts that will be made.

3. Preliminary Exploratory Analysis. The use of simple tools such as graphing and summary statistics to better understand the data. 

Review plots and summarize and note obvious temporal structures auch as trends, seasonality, anomalies such missing data, corruption, and outliers, and any other structures that may impact forecasting.

4. Choosing and Fitting Models. Evaluate two, three, or a suite of models of varying types on the problem. 

Models may be chosen for evaluation based on the assumptions they make and whether the dataset conforms. 

Models are configured and fit to the historical data.

5. Using and Evaluating a Forecasting Model. The model is used to make forecasts and the performance of those forecasts is evaluated and the skill of the models is estimated. 

This may involve back-testing with historical data or waiting for new observations to become available for comparison.

This 5-step process provides a strong overview from starting off with an idea or problem statement and leading to a model that can be used to make predictions.

The focus of the process is on understanding the problem and fitting a good model.



## Time Series Forecasting Explained

[Time Series From Scratch](https://towardsdatascience.com/time-series-analysis-from-scratch-seeing-the-big-picture-2d0f9d837329)

[An Ultimate Guide to Time Series Analysis in Pandas](https://towardsdatascience.com/an-ultimate-guide-to-time-series-analysis-in-pandas-d511b8e80e81)

How to analyze a time series? To answer this question, you’ll have to understand two fundamental concepts in time series — trend and seasonality.

As the name suggests, trend represents the general movement over time while seasonality represents changes in behavior in the course of a single season. 

For example, most monthly sampled data have yearly seasonality, meaning some patterns repeat in certain months every year, regardless of the trend.



## Time Series Forecasting with PyCaret

[Time Series Forecasting with PyCaret Regression Module](https://towardsdatascience.com/time-series-forecasting-with-pycaret-regression-module-237b703a0c63)

A step-by-step tutorial to forecast a single time series using PyCaret

## [Multiple Time Series Forecasting with PyCaret](https://towardsdatascience.com/multiple-time-series-forecasting-with-pycaret-bc0a779a22fe)

A step-by-step tutorial to forecast multiple time series using PyCaret


## [Error Metrics used in Time Series Forecasting](https://medium.com/analytics-vidhya/error-metrics-used-in-time-series-forecasting-modeling-9f068bdd31ca)

Some of the common error metrics used in particular with Time Series Forecasting model assessment and they are:

- Mean Square Error
- Root Mean Square Error
- Mean Absolute Error
- Mean Absolute Percentage Error
- Mean Frequency Error


## References

[How to Reframe Your Time Series Forecasting Problem](https://machinelearningmastery.com/reframe-time-series-forecasting-problem/)


[Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)


[How to Develop LSTM Models for Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)

[LSTM for time series prediction](https://towardsdatascience.com/lstm-for-time-series-prediction-de8aeb26f2ca)

[Predicting stock prices using Deep Learning LSTM model in Python](https://thinkingneuron.com/predicting-stock-prices-using-deep-learning-lstm-model-in-python/)

[Time Series Forecasting with Python 7-Day Mini-Course](https://machinelearningmastery.com/time-series-forecasting-python-mini-course/)

[Classical Time Series Forecasting Methods in Python (Cheat Sheet)](https://machinelearningmastery.com/time-series-forecasting-methods-in-python-cheat-sheet/)



[LSTMs for Human Activity Recognition Time Series Classification](https://machinelearningmastery.com/how-to-develop-rnn-models-for-human-activity-recognition-time-series-classification/)



