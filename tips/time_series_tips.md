# Time Series Tips

Data Preparation for Multivariate LSTM Forecasting

## Tips and Tricks (Multivariate time series)

- Do not shuffle train/test datasets
- Convert dataset to 3D supervised shape [samples, ntimesteps n_feature].
- Reframe the problem as supervised learning problem with  (X, y) datasets.

## [How to Check if Time Series Data is Stationary with Python?](https://gist.github.com/codecypher/80d6bc9ade74a33ae8b5ca863662896d)

## to_categorical

[Simple MNIST convnet](https://keras.io/examples/vision/mnist_convnet/)


## Keras Time Series Examples

You can also skim these two examples to see if u get any ideas:

[Timeseries forecasting for weather prediction](https://keras.io/examples/timeseries/timeseries_weather_forecasting/)

[Timeseries classification from scratch](https://keras.io/examples/timeseries/timeseries_classification_from_scratch/)

## Steps

  1. data_prep
  2. select_features
  3. train_test_split
  4. normalize
  5. Reshape input to be 3D [samples, timesteps, features]
  6. Save scalers to .pkl
  7. Save dataset to .npz

## Params Class

```py
class Params:
    """
    Model hyper-parameters
    """
    # stream = 'cse'     # Class variable
    def __init__(self):
        # Instance variables
        self.root_dir = "."
        self.data_dir = "../data/npz"
        self.output_dir = "../output"

        self.num_epochs = 100
        self.batch_size = 16
        self.train_test_pct = 0.8

        self.num_steps_in = 10
        self.num_features = 11
        self.num_steps_out = 5
        self.units = self.num_steps_in * self.num_features
        self.activation = 'relu'
        self.dropout = 0
        self.learning_rate = 0.01

        self.log_interval = 10

        json_path = os.path.join("../models", "keras_lstm_mse.json")
        self.dict_optuna = load_json(json_path)

    def __repr__(self):
        return "<{0} num_steps_in: {1} num_features: {2} num_steps_out: {3}>".format(
            self.__class__.__name__,
            self.num_steps_in,
            self.num_features,
            self.num_steps_out)

    def __str__(self):
        d_vars = vars(self)
        s_out = ""
        for key, value in d_vars.items():
            if key != "dict_optuna":
                s_out += str(key) + ": " + str(value) + "\n"

        return s_out

    def print(self):
        s_out = ""
        d_vars = vars(self)
        for key, value in d_vars.items():
            if key != "dict_optuna":
                print(key, value)

    def set_optuna(self, ticker, n_dataset):
        steps_dataset = "{0}-{1}".format(str(self.num_steps_in), str(n_dataset))
        self.batch_size = params.dict_optuna[ticker][steps_dataset]['batch_size']
        self.units = params.dict_optuna[ticker][steps_dataset]['units']
        self.activation = params.dict_optuna[ticker][steps_dataset]['activation']
        self.dropout = params.dict_optuna[ticker][steps_dataset]['dropout']
        self.learning_rate = params.dict_optuna[ticker][steps_dataset]['learning_rate']

# Usage
params = Params()
call_func(params)

```


## Train/Test Split

```py
def train_test_split(args, data, n_steps, n_features, train_split_pct):
    """
    Split the data into train and test datasets

    Args:
        args (Config): config settings
        data (np.array): array of data to process
    """
    # Find index of column by name
    # idx_target = df_filtered.columns.get_loc(target_col)

    n_rows = data.shape[0]

    # Split the training data into X_train and y_train datasets
    # Get the number of rows to train the model on
    n_train = math.ceil(n_rows * pct_split)

    # Frame as supervised learning
    reframed = series_to_supervised(data, n_steps, n_features)
    np_data = reframed.values

    # Split into train and test sets
    train = np_data[:n_train, :]
    test = np_data[n_train:, :]

    # Split into input and outputs
    n_obs = n_steps * n_features
    X_train, y_train = train[:, :n_obs], train[:, -n_features]
    X_test, y_test = test[:, :n_obs], test[:, -n_features]

    return (X_train, y_train), (X_test, y_test)
```

## Normalization

```py
def normalize(X_train, y_train, Xtest, y_test):
    """
    The transform must be estimated on the training dataset only
    then applied to val/test datasets.

    Args:
        X_train (np.array): array of train input features
        y_train (np.array): array of train target feature
        X_test (np.array): array of test input features
        y_test (np.array): array of test target feature

    Returns:
        X_train_scaled (np.array): scaled array of train input features
        y_train_scaled (np.array): scaled array of train target feature
        X_test_scaled (np.array): scaled array of test input features
        y_test_scaled (np.array): scaled array of test target feature
        scaler (MinMaxScaler): scaler for input features
        scaler_pred (MinMaxScaler): scaler for target feature
    """
    # Normalize input features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    # Normalize target features
    scaler_pred = MinMaxScaler()
    y_train_scaled = scaler_pred.fit_transform(y_train)
    y_test_scaled = scaler_pred.transform(y_test)

    # Reshape input to be 3D [samples, timesteps, features]
    X_train_scaled = X_train.reshape((X_train.shape[0], n_steps, n_features))
    X_test_scaled = X_test.reshape((X_test.shape[0], n_steps, n_features))

    # # Save scaler to file
    # with open('scaler.pkl', 'wb') as file:
    #     pickle.dump(scaler, file)
    # with open('scaler_pred.pkl', 'wb') as file:
    #     pickle.dump(scaler_pred, file)

    return (X_train_scaled, y_train_scaled), (X_test_scaled, y_test_scaled), (scaler, scaler_pred)
```

## Evaluate Performance

```py
def compute_accuracy(args, X, yhat, y, scaler_pred):
    n_steps_in = args.num_steps_in
    n_features = args.num_features

    if len(yhat) != len(y):
        print("Assert error: predicted and y must be of equal length")
        return

    print(X.shape)
    
    # May be optional (depends on use case)
    X = X.reshape((X.shape[0], n_steps_in * n_features))
    y = y.reshape((len(y), 1))

    inv_yhat = np.concatenate((yhat, X[:, -(n_features - 1):]), axis=1)
    inv_yhat = scaler_pred.inverse_transform(inv_yhat)

    inv_y = np.concatenate((y, X[:, -(n_features - 1):]), axis=1)
    inv_y = scaler_pred.inverse_transform(inv_y)

    # Accuracy of predictions on datasets
    acc = 100 - (100 * (abs(inv_yhat - inv_y) / inv_y)).mean()

    return acc
```

```py
    yhat_test = model.predict(X_test)
    acc_test = compute_accuracy(params, X_test, yhat_test, y_test, scaler_pred)

```

### Time Series Tips and Tricks

[5 Tips for Working With Time Series in Python](https://medium.com/swlh/5-tips-for-working-with-time-series-in-python-d889109e676d)

[4 Common Machine Learning Data Transforms for Time Series Forecasting](https://machinelearningmastery.com/machine-learning-data-transforms-for-time-series-forecasting/)

[Training Time Series Forecasting Models in PyTorch](https://towardsdatascience.com/training-time-series-forecasting-models-in-pytorch-81ef9a66bd3a)

