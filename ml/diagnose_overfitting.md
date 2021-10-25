<head>
  <link rel="stylesheet" type="text/css" href="../css/style.css">
</head>

# How to Diagnose Overfitting and Underfitting?

Here are some notes on analyzing overfitting and underfitting.

## Why is my validation loss lower than my training loss?

In this tutorial, you will learn the three primary reasons your validation loss may be lower than your training loss when training your own custom deep neural networks.

At the most basic level, a loss function quantifies how “good” or “bad” a given predictor is at classifying the input data points in a dataset.

The smaller the loss, the better a job the classifier is at modeling the relationship between the input data and the output targets.

However, there is a point where we can overfit our model — by modeling the training data too closely, our model loses the ability to generalize.

Thus, we seek to:

- Drive our loss down, improving our model accuracy.

- Do so as fast as possible and with as little hyperparameter updates/experiments.

- Avoid overfitting our network and modeling the training data too closely.

It is a balancing act and our choice of loss function and model optimizer can dramatically impact the quality, accuracy, and generalizability of our final model.

Typical loss functions (objective functions or scoring functions) include:

- Binary cross-entropy
- Categorical cross-entropy
- Sparse categorical cross-entropy
- Mean Squared Error (MSE)
- Mean Absolute Error (MAE)
- Standard Hinge
- Squared Hinge

For most tasks:

- Loss measures the “goodness” of your model
- The smaller the loss, the better
- But careful not to overfit

### Reason 1: Regularization applied during training but not during validation/testing


When training a deep neural network we often apply **regularization** to help our model:

- Obtain higher validation/testing accuracy
- To generalize better to the data outside the validation and testing sets

Regularization methods often **sacrifice training accuracy to improve validation/testing accuracy** — in some cases this can lead to your validation loss being lower than your training loss.

Also keep in mind that regularization methods such as dropout are not applied at validation/testing time.


### Reason 2: Training loss is measured during each epoch while validation loss is measured after each epoch

The second reason you may see validation loss lower than training loss is due to how the loss value is measured and reported:

- Training loss is measured _during_ each epoch
- While validation loss is measured _after_ each epoch

Your training loss is continually reported over the course of an entire epoch, but **validation metrics are computed over the validation set only once the current training epoch is completed**. 

Thus, on average the training losses are measured half an epoch earlier.

### Reason 3: The validation set may be easier than the training set (or there may be leaks)

The third common reason is due to the data distribution itself.

Consider how your validation set was acquired:

- Can you guarantee that the validation set was sampled from the same distribution as the training set?

- Are you certain that the validation examples are just as challenging as your training images?

- Can you assure there was no _data leakage_ (training samples getting accidentally mixed in with validation/testing samples)?

- Are you confident your code created the training, validation, and testing splits properly?

Every deep learning practitioner has made the above mistakes at least once in their career.


—————



## Diagnose Overfitting and Underfitting of LSTM Models

It can be difficult to determine whether your Long Short-Term Memory model is performing well on your sequence prediction problem.

You may be getting a good model skill score but it is important to know whether your model is a good fit for your data or if it is underfit or overfit and could do better with a different configuration.

In this tutorial, you will discover how you can diagnose the fit of your LSTM model on your sequence prediction problem.

After completing this tutorial, you will know:

- How to gather and plot training history of LSTM models.

- How to diagnose an underfit, good fit, and overfit model.

- How to develop more robust diagnostics by averaging multiple model runs.

### Tutorial Overview

This tutorial is divided into 6 parts; they are:

1. Training History in Keras
2. Diagnostic Plots
3. Underfit Example
4. Good Fit Example
5. Overfit Example
6. Multiple Runs Example


### Training History in Keras

You can learn a lot about the behavior of your model by reviewing its performance over time.

LSTM models are trained by calling the **fit()** function which returns a variable called _history_ that contains a trace of the loss and any other metrics specified during the compilation of the model. 

Theae metric scores are recorded at the end of each epoch.

### Diagnostic Plots

The training history of your LSTM models can be used to diagnose the behavior of your model.

**Learning Curve:** Line plot of learning (y-axis) over experience (x-axis).

During the training of a machine learning model, the current state of the model at each step of the training algorithm can be evaluated. It can be evaluated on the training dataset to give an idea of how well the model is learning. 

You can plot the performance of your model using the `matplotlib` library. 

- **Train Learning Curve:** Learning curve calculated from the training dataset that gives an idea of how well the model is learning.

- **Validation Learning Curve:** Learning curve calculated from a hold-out validation dataset that gives an idea of how well the model is generalizing.

It is common to create learning curves for multiple metrics such as in the case of classification predictive modeling problems, where the model may be optimized according to cross-entropy loss and model performance is evaluated using classification accuracy. 

In this case, two plots are created: one for the learning curves of each metric where each plot can show two learning curves (one for each of the train and validation datasets). 

- **Optimization Learning Curves:** Learning curves calculated on the metric by which the parameters of the model are being optimized such as loss.

- **Performance Learning Curves:** Learning curves calculated on the metric by which the model will be evaluated and selected such as accuracy.


### Underfit Example

An underfit model is one that is demonstrated to perform well on the training dataset and poor on the test dataset.

This can be diagnosed from a plot where the training loss is lower than the validation loss, and the validation loss has a trend that suggests further improvements are possible.

<div class="image-preview">
    <img width="600" alt="Plot showing underfit model" src="https://machinelearningmastery.com/wp-content/uploads/2017/07/Diagnostic-Line-Plot-Showing-an-Underfit-Model.png" />
</div>

<div class="image-preview">
    <img width="600" alt="Plot of Training Learning Curve of Underfit Model That Does Not Have Sufficient Capacity" src="https://machinelearningmastery.com/wp-content/uploads/2019/02/Example-of-Training-Learning-Curve-Showing-An-Underfit-Model-That-Does-Not-Have-Sufficient-Capacity.png" />
</div>

<div class="image-preview">
    <img width="600" alt="Plot of Training Learning Curve Showing an Underfit Model That Requires Further Training" src="https://machinelearningmastery.com/wp-content/uploads/2018/12/Example-of-Training-Learning-Curve-Showing-An-Underfit-Model-That-Requires-Further-Training.png" />
</div>

A plot of learning curves shows underfitting if:

- The training loss remains flat regardless of training.

- The training loss continues to decrease until the end of training.

### Good Fit Example

A good fit is a case where the performance of the model is good on both the train and validation sets.

This can be diagnosed from a plot where the train and validation loss decrease and stabilize around the same point.

<div class="image-preview">
    <img width="600" alt="Plot showing good fit" src="https://machinelearningmastery.com/wp-content/uploads/2017/07/Diagnostic-Line-Plot-Showing-a-Good-Fit-for-a-Model.png" />
</div>

<div class="image-preview">
    <img width="600" alt="Plot of Train and Validation Learning Curves Showing a Good Fit" src="https://machinelearningmastery.com/wp-content/uploads/2018/12/Example-of-Train-and-Validation-Learning-Curves-Showing-A-Good-Fit.png" />
</div>

### Overfit Example

An overfit model is one where performance on the train set is good and continues to improve whereas performance on the validation set improves to a point and then begins to degrade.

This can be diagnosed from a plot where the train loss slopes down and the validation loss slopes down, hits an inflection point, and starts to slope up again.

<div class="image-preview">
    <img width="600" alt="Plot showing overfit model" src="https://machinelearningmastery.com/wp-content/uploads/2017/07/Diagnostic-Line-Plot-Showing-an-Overfit-Model.png" />
</div>

<div class="image-preview">
    <img width="600" alt="Plot of Train and Validation Learning Curves Showing an Overfit Model" src="https://machinelearningmastery.com/wp-content/uploads/2018/12/Example-of-Train-and-Validation-Learning-Curves-Showing-An-Overfit-Model.png" />
</div>

### Multiple Runs Example

LSTMs are stochastic which means that you will get a different diagnostic plot each run.

It can be useful to repeat the diagnostic run multiple times (say 5, 10, or 30). 

The train and validation traces from each run can then be plotted to give a more robust idea of the behavior of the model over time.

The example below runs the same experiment a number of times before plotting the trace of train and validation loss for each run.

## Diagnosing Unrepresentative Datasets

Learning curves can also be used to diagnose properties of a dataset and whether it is relatively representative.

An unrepresentative dataset means a dataset that may not capture the statistical characteristics relative to another dataset drawn from the same domain, such as between a train and a validation dataset. This can commonly occur if the number of samples in a dataset is too small, relative to another dataset.

There are two common cases that could be observed; they are:

- Training dataset is relatively unrepresentative.

- Validation dataset is relatively unrepresentative.

### Unrepresentative Train Dataset

An unrepresentative training dataset means that the training dataset does not provide sufficient information to learn the problem, relative to the validation dataset used to evaluate it.

This may occur if the training dataset has too few examples as compared to the validation dataset.

This situation can be identified by a learning curve for training loss that shows improvement and similarly a learning curve for validation loss that shows improvement, but a large gap remains between both curves.

<div class="image-preview">
    <img width="600" alt="Plot of Train and Validation Learning Curves Showing a Training Dataset That May Be too Small Relative to the Validation Dataset" src="https://machinelearningmastery.com/wp-content/uploads/2018/12/Example-of-Train-and-Validation-Learning-Curves-Showing-a-Training-Dataset-the-May-be-too-Small-Relative-to-the-Validation-Dataset.png" />
</div>

### Unrepresentative Validation Dataset

An unrepresentative validation dataset means that the validation dataset does not provide sufficient information to evaluate the ability of the model to generalize.

This may occur if the validation dataset has too few examples as compared to the training dataset.

This case can be identified by a learning curve for training loss that looks like a good fit (or other fits) and a learning curve for validation loss that shows noisy movements around the training loss.

<div class="image-preview">
    <img width="600" alt="Plot of Train and Validation Learning Curves Showing a Validation Dataset That May Be too Small Relative to the Training Dataset" src="https://machinelearningmastery.com/wp-content/uploads/2018/12/Example-of-Train-and-Validation-Learning-Curves-Showing-a-Validation-Dataset-the-May-be-too-Small-Relative-to-the-Training-Dataset.png" />
</div>

This may also be identified by a validation loss that is lower than the training loss which indicates that the validation dataset may be easier for the model to predict than the training dataset.

<div class="image-preview">
    <img width="600" alt="Plot of Train and Validation Learning Curves Showing a Validation Dataset That Is Easier to Predict Than the Training Dataset" src="https://machinelearningmastery.com/wp-content/uploads/2018/12/Example-of-Train-and-Validation-Learning-Curves-Showing-a-Validation-Dataset-that-is-Easier-to-Predict-than-the-Training-Dataset.png" />
</div>

## References

[How to Diagnose Overfitting and Underfitting of LSTM Models](https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/)

[How to use Learning Curves to Diagnose Machine Learning Model Performance](https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/)

[Why is my validation loss lower than my training loss?](https://www.pyimagesearch.com/2019/10/14/why-is-my-validation-loss-lower-than-my-training-loss/)

[How to Mitigate Overfitting with K-Fold Cross-Validation using sklearn](https://towardsdatascience.com/how-to-mitigate-overfitting-with-k-fold-cross-validation-518947ed7428)

