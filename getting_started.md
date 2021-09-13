# Getting Started

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


## References

[Gettting Started with Machine Learning](https://machinelearningmastery.com/start-here/)

[Multinomial Logistic Regression With Python](https://machinelearningmastery.com/multinomial-logistic-regression-with-python/)

[Multinomial Logistic Regression In a Nutshell](https://medium.com/ds3ucsd/multinomial-logistic-regression-in-a-nutshell-53c94b30448f)

[One-vs-Rest and One-vs-One for Multi-Class Classification](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)

