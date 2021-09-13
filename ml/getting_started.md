# Machine Learning Tips

## [How to Select an ML Model?](https://www.kdnuggets.com/2021/08/select-initial-model-data-science-problem.html)

There are many models to choose from with seemingly endless variants.

There are usually only slight alterations needed to change a regression model into a classification model and vice versa. 

This work has already been done for you with the standard python supervised learning packages, so you only need to select what option you want.


## Why Simple Models?

[Regression for Classification](https://towardsdatascience.com/regression-for-classification-hands-on-experience-8754a909a298)

- Linear Regression
- Logistic Regression

Why should you start with these simple models? It is likely that your problem does not need anything fancy.

Busting out some deep learning model and spending hundreds on AWS fees to get only a slight accuracy bump is not worth it.

These two models have been studied for decades and are some of the most well-understood models in machine learning.

They are easily interpretable. Both models are linear, so their inputs translate to their output in a way that you could calculate by hand.

### Save yourself some headache.
   
Even if you are an experienced data scientist, you should still know the performance of these models on your problem since they are so effortless to implement and test.


To covert regression problem to classificatiom problem, there are 2 great solutions.

- Logistic Regression: For binary classification

- Softmax Regression: For multi class classification


## Multinomial Logistic Regression (MLR)

Multinomial Logistic Regression is a classification algorithm used to do multiclass classification.

Multinomial logistic regression is an extension of logistic regression that adds support for multi-class classification problems.

sklearn: We can fit a multinomial logistic regression with L1 penalty on a subset of the MNIST digits classification task.

The primary assumptions of a linear regression, multiple and singular, are:

1. Linearity: There is a linear relationship between the outcome and predictor variable(s).

2. Normality: The residuals, or errors, calculated by subtracting the predicted value from the actual value, follow a normal distribution.

3. Homoscedasticity: The variability in the dependent variable is equal for all values of the independent variable(s).


We often still construct linear models for data that does not quite meet these standards, with many independent variables as in MLR, we run into other problems such multicollinearity where variables that are supposed to be independent vary with each other, and the presence of categorical variables such as an ocean temperature being classified as cool, warm, or hot instead of quantified in degrees. 


When your MLR models get complicated as you take steps to improve their efficacy, avoid trying to use coefficients to interpret changes in the outcome against changes in individual predictors. 

Create predictions while varying a sole predictor and observe how the prediction changes, and use the nature of these changes to form your conclusions.



## [What aspect ratio to use for line plots](https://towardsdatascience.com/should-you-care-about-the-aspect-ratio-when-creating-line-plots-ed423a5dceb3)

One of the most overlooked aspects of creating charts is the use of correct aspect ratios. 

### Calculating the aspect ratio

The concept of banking to 45 degrees is used to have coherency between the information presented and information perceived. 

Thus, we need to make sure that the orientation of the line segments in the chart is as close as possible to a slope of 45 degrees.

Here, the median absolute slope banking method has been used to calculate the aspect ratio for the sunspots plot. ￼

The ggthemes package provides a function called bank_slopes() to calculate the aspect ratio of the plot which takes x and y values as the two arguments. The default method is the median absolute slope banking. 

### Best practices:

- **Plotting multiple line graphs for comparison on a single chart:** The default aspect ratio works only if you do not plan to compare two different plots.


- **Comparing different line graphs from different charts:** Make sure the aspect ratio for each plot remains the same. Otherwise, the visual interpretation will be skewed. 

  1. Using incorrect or default aspect ratios: In this case, we choose the aspect ratios such that the plots end up being square-shaped.

  2. Calculating aspect ratios per plot: The best approach to compare the plots is to calculate the aspect ratios for each plot. 

**Time-series:** It is best to calculate the aspect ratio since some hidden information can be more pronounced when using the correct aspect ratio for the plot.


## Deep Learning

[Deep Learning (Keras)](https://machinelearningmastery.com/start-here/#deeplearning)

[Better Deep Learning Performance](https://machinelearningmastery.com/start-here/#better)

[How To Improve Deep Learning Performance](https://machinelearningmastery.com/improve-deep-learning-performance/)

[How to Choose Loss Functions When Training Deep Learning Neural Networks](https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/)

[Understand the Impact of Learning Rate on Neural Network Performance](https://machinelearningmastery.com/understand-the-dynamics-of-learning-rate-on-deep-learning-neural-networks/)


## Challenge of Training Deep Learning Neural Networks

[A Gentle Introduction to the Challenge of Training Deep Learning Neural Network Models](https://machinelearningmastery.com/a-gentle-introduction-to-the-challenge-of-training-deep-learning-neural-network-models/)

### Neural Networks Learn a Mapping Function

Deep learning neural networks learn a _mapping function_. 

Developing a model requires historical data from the domain that is used as training data which is comprised of observations or examples from the domain with input elements that describe the conditions and an output element that captures what the observation means.

- A problem where the output is a quantity would be described generally as a regression predictive modeling problem. 

- A problem where the output is a label would be described generally as a classification predictive modeling problem.

A neural network model uses the examples to learn how to map specific sets of input variables to the output variable. 

The NN must learn the mapping in such a way that this mapping works well for the training dataset but also works well on new examples not seen by the model during training which is called _generalization_. 


We can describe the relationship between the input variables and the output variables as a complex mathematical function. 

For a given model problem, we must believe that a true mapping function exists to best map input variables to output variables and that a neural network model can do a reasonable job at approximating the true unknown underlying mapping function.

Thus, we can describe the broader problem that neural networks solve as _function approximation_. 

A NN learns to approximate an unknown underlying mapping function given a training dataset by learning weights and the model parameters, given a specific network structure that we design.

However, finding the parameters for neural networks in general is hard.

In fact, training a neural network is the most challenging part of using the technique.

The use of nonlinear activation functions in the neural network means that the optimization problem that we must solve in order to find model parameters is not convex.

Solving this optimization is challenging, not least because the error surface contains many local optima, flat spots, and cliffs.


### Navigating the Non-Convex Error Surface

A NN model has a specific set of weights can be evaluated on the training dataset and the average error over all training datasets can be thought of as the error of the model. 

A change to the model weights will result in a change to the model error. Therefore, we seek a set of weights that result in a model with a small error.

The process involves repeating the steps of evaluating the model and updating the model parameters in order to step down the error surface which is repeated until a set of parameters is found that is good enough or the search process gets stuck.

Thus, the process is a search or an optimization and we refer to optimization algorithms that operate in this way as gradient optimization algorithms sonce they naively follow along the error gradient. In practice, this is more art than science.

The algorithm that is most commonly used to navigate the error surface is called stochastic gradient descent (SGD).

Other global optimization algorithms designed for non-convex optimization problems could be used, such as a genetic algorithm but stochastic gradient descent is more efficient since it uses the gradient information specifically to update the model weights via an algorithm called _backpropagation_.

Backpropagation refers to a technique from calculus to calculate the derivative (e.g. the slope or the gradient) of the model error for specific model parameters which allows the model weights to be updated to move down the gradient.

### Components of the Learning Algorithm

Training a deep learning neural network model using stochastic gradient descent with backpropagation involves choosing a number of components and hyperparameters:

- Loss Function: The function used to estimate the performance of a model with a specific set of weights on examples from the training dataset.

- Weight Initialization: The procedure by which the initial small random values are assigned to model weights at the beginning of the training process.

- Batch Size: The number of examples used to estimate the error gradient before updating the model parameters.

- Learning Rate: The amount that each model parameter is updated per cycle of the learning algorithm.

- Epochs. The number of complete passes through the training dataset before the training process is terminated.


### Decrease Neural Network Size and Maintain Accuracy

[Decrease Neural Network Size and Maintain Accuracy](https://towardsdatascience.com/decrease-neural-network-size-and-maintain-accuracy-knowledge-distillation-6efb43952f9d)

Some neural networks are too big to use. There is a way to make them smaller but keep their accuracy.

1. Pruning

2. Knowledge distillation


## References

[Gettting Started with Machine Learning](https://machinelearningmastery.com/start-here/)

[Multinomial Logistic Regression With Python](https://machinelearningmastery.com/multinomial-logistic-regression-with-python/)

[Multinomial Logistic Regression In a Nutshell](https://medium.com/ds3ucsd/multinomial-logistic-regression-in-a-nutshell-53c94b30448f)

[One-vs-Rest and One-vs-One for Multi-Class Classification](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)


