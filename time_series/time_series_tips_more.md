# Working with Time Series

<!-- MarkdownTOC -->

- 5 Tips for Working With Time Series in Python
    - Removing noise with the Fourier Transform
    - Removing noise with the Kalman Filter
    - Dealing with Outliers
    - The right way to normalize time series data.
    - A flexible way to compute returns.
- 4 Common Machine Learning Data Transforms for Time Series Forecasting
    - Transforms for Time Series Data
        - Power Transform
        - Difference Transform
        - Standardization
        - Normalization
    - Considerations for Model Evaluation
    - Order of Data Transforms
- Training Time Series Forecasting Models in PyTorch
    - Data Quality/Preprocessing
    - Model Choice and hyper-parameter selection
    - Robustness

<!-- /MarkdownTOC -->


# [5 Tips for Working With Time Series in Python](https://medium.com/swlh/5-tips-for-working-with-time-series-in-python-d889109e676d)

## Removing noise with the Fourier Transform

## Removing noise with the Kalman Filter

## Dealing with Outliers

Outliers are usually undesirable because they affect our conclusions if we are not careful when dealing with them. For example, the Pearson correlation formula can have a very different result if there are large enough outliers in our data.

Outlier analysis and filtering in time series requires a more sophisticated approach than in normal data because **we cannot use future information to filter past outliers**.

One quick way to remove outliers is doing it in a rolling/expanding basis.

A common algorithm to find outliers is to compute the mean and standard deviation of the data and check which values are _n_ standard deviations above or below the mean (typically, n = 3). Those values are then marked as outliers.

NOTE: This particular approach will usually work better if you standardize your data (and it is conceptually more correct to use it that way).


## The right way to normalize time series data.

Many posts use the classical fit-transform approach with time series as if they could be treated as normal data. As with outliers, you cannot use future information to normalize data from the past unless you are 100% sure the values you are using to normalize are constant over time.

The right way to normalize time series is in a **rolling/expanding** basis.

I used the Sklearn API to create a class  to normalize data avoiding look-ahead bias. Since it inherits `BaseEstimator` and `TransformerMixin`, it is possible to embed this class in a Sklearn pipeline.

## A flexible way to compute returns.

The last tip is focused on quantitative analysis of financial time series.

When working with returns, it is usually necessary to have a basic framework to quickly compute log and arithmetic returns in different periods of time.

When filtering financial time series, the ideal procedure filters returns first and then goes back to prices. So you are free to add this step to the code from section 4.


----------



# [4 Common Machine Learning Data Transforms for Time Series Forecasting](https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/)

## Transforms for Time Series Data

Given a univariate time series dataset, there are four transforms that are popular when using machine learning methods to model and make predictions:

### Power Transform

A _power transform_ removes a shift from a data distribution to make the distribution more normal (Gaussian).

On a time series dataset, this can have the effect of removing a change in variance over time.

Popular examples are the the log transform (positive values) or generalized versions such as the Box-Cox transform (positive values) or the Yeo-Johnson transform (positive and negative values).

### Difference Transform

A _difference transform_ is a simple way for removing a systematic structure from the time series.

For example, a trend can be removed by subtracting the previous value from each value in the series which is called _first order differencing_. The process can be repeated (difference the differenced series) to remove second order trends, and so on.

A seasonal structure can be removed in a similar way by subtracting the observation from the prior season, say 12 time steps ago for monthly data with a yearly seasonal structure.

A single differenced value in a series can be calculated with a custom function named ``difference()`` shown below which takes the time series and the interval for the difference calculation, say 1 for a trend difference or 12 for a seasonal difference.

### Standardization

Standardization is a transform for data with a Gaussian distribution that subtracts the mean and divides the result by the standard deviation of the data sample which has the effect of transforming the data to have mean of zero (or centered) and a standard deviation of 1.

This resulting distribution is called a _standard Gaussian distribution_ or _standard normal_, hence the name of the transform.

We can perform standardization using the `StandardScaler` object in Python from the scikit-learn library.

This class allows the transform to be fit on a training dataset by calling `fit()`, applied to one or more datasets (train and test) by calling `transform()` and also provides a function to reverse the transform by calling `inverse_transform()`.

```py
    from sklearn.preprocessing import StandardScaler
    from numpy import array

    # define dataset
    data = [x for x in range(1, 10)]
    data = array(data).reshape(len(data), 1)
    print(data)

    # fit transform
    transformer = StandardScaler()
    transformer.fit(data)

    # difference transform
    transformed = transformer.transform(data)
    print(transformed)

    # invert difference
    inverted = transformer.inverse_transform(transformed)
    print(inverted)
```

### Normalization

Normalization is a rescaling of data from the original range to a new range between 0 and 1.

As with standardization, this can be implemented using a transform object from the scikit-learn library called the `MinMaxScaler` class. In addition to normalization, this class can be used to rescale data to any range you wish by specifying the preferred range in the constructor of the object.

MinMaxScaler can be used in the same way to fit, transform, and inverse the transform.

```py
    from sklearn.preprocessing import MinMaxScaler
    from numpy import array

    # define dataset
    data = [x for x in range(1, 10)]
    data = array(data).reshape(len(data), 1)
    print(data)

    # fit transform
    transformer = MinMaxScaler()
    transformer.fit(data)

    # difference transform
    transformed = transformer.transform(data)
    print(transformed)

    # invert difference
    inverted = transformer.inverse_transform(transformed)
    print(inverted)
```

## Considerations for Model Evaluation

We have mentioned the importance of being able to invert a transform on the predictions of a model in order to calculate a model performance statistic that is directly comparable to other methods.

Another concern is the problem of **data leakage**.

Three of the above data transforms estimate coefficients from a provided dataset that are then used to transform the data:

  - Power Transform: lambda parameter
  - Standardization: mean and standard deviation statistics
  - Normalization: min and max values

**These coefficients must be estimated on the training dataset only.**

```py
    scaler = MinMaxScaler()
    X_train_arr = scaler.fit_transform(X_train)
    X_val_arr = scaler.transform(X_val)
    X_test_arr = scaler.transform(X_test)

    y_train_arr = scaler.fit_transform(y_train),
    y_val_arr = scaler.transform(y_val)
    y_test_arr = scaler.transform(y_test)
```

Once estimated, the transform can be applied using the coefficients to the training and the test dataset before evaluating your model.

If the coefficients are estimated using the entire dataset prior to splitting into train and test sets, then there is a _small leakage_ of information from the test dataset to the training dataset which can result in estimates of model skill that are optimistically _biased_.

Thus, you may want to enhance the estimates of the coefficients with domain knowledge such as expected min/max values for all time in the future.

Generally, differencing does not suffer from the same problems. In most cases, such as one-step forecasting, the lag observations are available to perform the difference calculation. If not, the lag predictions can be used wherever needed as a proxy for the true observations in difference calculations.

## Order of Data Transforms

You may want to experiment with applying multiple data transforms to a time series prior to modeling.

It is quite common to;

- apply a power transform to remove an increasing variance
- apply seasonal differencing to remove seasonality and 
- apply one-step differencing to remove a trend.

The order that the transform operations are applied is important.

Intuitively, we can think through how the transforms may interact.

  - Power transforms should probably be performed prior to differencing.

  - Seasonal differencing should be performed prior to one-step differencing.

  - Standardization is linear and should be performed on the sample after any nonlinear transforms and differencing.

  - Normalization is a linear operation but it should be the final transform performed to maintain the preferred scale.

As such, a suggested ordering for data transforms is as follows:

  1. Power Transform
  2. Seasonal Difference
  3. Trend Difference
  4. Standardization
  5. Normalization

Obviously, you would only use the transforms required for your specific dataset.

It is important to remember that when the transform operations are inverted, the order of the inverse transform operations must be reversed. Specifically, the inverse operations must be performed in the following order:

  1. Normalization
  2. Standardization
  3. Trend Difference
  4. Seasonal Difference
  5. Power Transform


----------


# [Training Time Series Forecasting Models in PyTorch](https://towardsdatascience.com/training-time-series-forecasting-models-in-pytorch-81ef9a66bd3a)

Lessons and tips learned from training hundreds of PyTorch time series forecasting models in many different domains.

Before getting started, you should determine if your problem is actually a forecasting problem since this will guide how you should proceed.

Sometimes it might better to cast a forecasting problem as a classification problem. For example, if the exact number forecasted is not that important you could bucket it into ranges then use a classification model.

In addition, you should have some understanding of deployment and what the end product will look like. If you require millisecond latency for stock trading then a huge transformer model with 20 encoder layers probably will not function no matter what your test MAE is.

- **Anomaly detection:** A general technique to detect outliers in time series data.

  Anomalies usually form a very small part of the dataset and are substantially different from other data points. Thus, anomaly detection can be seen as an extreme form of binary classification but it is usually treated as a separate area.

  Most anomaly detection methods are unsupervised since we are often unlikely to recognize anomalies until they occur. See this paper for more information.

- **Time Series Classification:** This is similar to other forms of classification where we take a temporal sequence and want to classify it into a number of categories. Unlike anomaly detection, we generally have a more balanced number of examples of each class (though it may still be skewed something like 10%, 80%, 10%).

- **Time Series Forecasting:** In forecasting, we generally want to predict the next value or the next (n) values in a sequence of temporal data which is what this article will focus on.

- **Time Series Prediction:** This term is ambiguous and could mean many things. Most people usually use it to refer to either forecasting or classification in this context.

- **Time Series Analysis:** A general umbrella term that can include all of the above. However, I usually associate it more with just looking over time series data and comparing different temporal structures than inherently designing a predictive model. 

  For example, if you did develop a time series forecasting model than it could possibly tell you more about the casual factors in your time series and enable more time series analysis.

## Data Quality/Preprocessing

- **Always scale or normalize data:** Scaling or normalizing your data improves performance in 99% of uses cases. 

  Unless you have very small values then this is a step you should always take. 
  
  Flow Forecast has built in scalers and normalizers that are easy to use. 
  
  Failure to scale your data can often cause the loss to explode especially when training some transformers.

- **Double check for null, improperly encoded, or missing values:** Sometimes missing values are encoded in a weird way. 

  For example, some weather stations encode missing precip values as -9999 which can cause a lot of problems as a regular NA check will not catch this. 
  
  Flow forecast provided a module for interpolating missing values and warning about possibly incorrectly entered data.

- **Start with a fewer number of features:** In general, it is easier to start with fewer features and add more in, depending on performance.

## Model Choice and hyper-parameter selection

- **Visualize time lags to determine forecast_history:** In time series forecasting, regardless of model we have the number of time-steps that we want to pass into the model which will vary somewhat with architecture as some models are better at learning long range dependencies, but finding an initial range is useful. 

  In some cases, really long term dependencies might not be useful at all.

- **Start with DA-RNN:** The DA-RNN model creates a very strong time series baseline. 

  Usually transformers can outperform it, but they usually require more data and more careful hyper-parameter tuning. 

  Flow forecast provides an easy to use implementation of DA-RNN.

- **Determining a length to forecast:** The number of time steps your model forecasts at once is a tricky hyper-parameter to determine what values to search. 

  You can still generate longer forecasts but you do this by appending the previous forecasts. 
  
  If your goal is to predict to long range of time steps then you may want them directly weighed into the loss function, but having too many time steps at once can confuse the model. 
  
  In most of the hyper-parameter sweeps I have found a shorter forecast length works well.

- **Start with a low learning rate:** It is best to pick a low learning rate for most time series forecasting models.

- **Adam is not always the best:** Sometimes other optimizers can work better. 

  For example, `BertAdam` is good for transformer type models whereas for DA-RNN vanilla can work well.

## Robustness

- **Simulate and run play by play analysis of different scenarios:** 

  Flow Forecast makes it easy to simulate your model performance under different conditions. 
  
  For example, if you are forecasting stream flows you might try inputting really large precipitation values and see how the model responds.

- **Double check heatmaps and other interpretability metrics:** 

  Sometimes you may look at a model and think it is performing well, then you check the heatmap and see the model is not using the important features for forecasting. 
  
  When you perform further testing, it may become obvious that the model was just learning to memorize rather than the actual casual effects of features.



