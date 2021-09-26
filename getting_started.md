# Getting Started

## How to Select an ML Model?

Every new data scientist finds that they need to decide what that  what model to use for a problem.

There are many models to choose from with seemingly endless variants.

There are usually only slight alterations needed to change a regression model into a classification model and vice versa.

There are a lot of models to choose from, so consider starting with regression models to obtain a simple baseline for comparison. 

The model you choose should do better than simple regression. 

This work has already been done for you with the standard python supervised learning packages, so you only need to select what option you want.


## Why Simple Models?

The two most common regression algorithms are:

- Linear Regression
- Logistic Regression

Why should you start with these simple models? It is likely that your problem does not need anything fancy.

Busting out some deep learning model and spending hundreds on AWS fees to get only a slight accuracy bump is not worth it.

These two models have been studied for decades and are some of the most well-understood models in machine learning.

They are easily interpretable. Both models are linear, so their inputs translate to their output in a way that you could calculate by hand.

### Save yourself some headache.
   
Even if you are an experienced data scientist, you should still know the performance of these models on your problem since they are so effortless to implement and test.

To convert regression problem to classificatiom problem, there are two common solutions:

- Logistic Regression: binary classification

- Softmax Regression: multiclass classification


## Multinomial Logistic Regression (MLR)

Multinomial Logistic Regression is a classification algorithm used to do multiclass classification.

Multinomial logistic regression is an extension of logistic regression that adds support for multi-class classification problems.

sklearn: We can fit a multinomial logistic regression with L1 penalty on a subset of the MNIST digits classification task.

The three primary assumptions of linear regression are:


We often still construct linear models for data that does not quite meet these standards

With many independent variables as in MLR, we run into other problems such multicollinearity where variables that are supposed to be independent vary with each other, and the presence of categorical variables such as an ocean temperature being classified as cool, warm or hot instead of quantified in degrees. 


When your MLR models get complicated as you take steps to improve their efficacy, avoid trying to use coefficients to interpret changes in the outcome against changes in individual predictors. 

Create predictions while varying a sole predictor and observe how the prediction changes, and use the nature of these changes to form your conclusions.


## Understand Machine Learning Algorithms

You need to know what algorithms are available for a given problem, how they work, and how to get the most out of them. However, this does not mean you need to hand-code the algorithms from scratch.

In fact, there are many other concepts that are much more important when implmenting ML models:

- How to properly define an ML problem?
- How to select a dataset?
- How to perform data preparation?
- How to design an ML solution?
- How to properly train an ML model?
- How to deploy and ML solution?

- Parametric vs Nonparametric Algorithms
- Supervised vs Unsupervised Algorithms
- The Bias-Variance Trade-Off
- Data Preparation Techniques
- Feature Engineering Techniques

- How to Diagnose/Fix Overfitting and Underfitting?
- How to create a data pipeline?

- How to deal with small data?
- How to deal with imbalanced dataset?
- How to choose a Performance/Error Metric?
- How to debug ML models?

- Anomaly Detection
- AutoML

Even if you are an experienced data scientist, you should still know the performance of simpler models on your dataset/problem since they are so effortless to implement and test. In fact, it is considered a best practice to run test datasets to see if your chosen machine learning model outperforms a recognised benchmark.

### Feature Engineering Tools

Feature engineering techniques for machine learning are a fundamental topic in machine learning but one that is often overlooked or deceptively simple.

There are many tools that will help you in automating the entire feature engineering process and producing a large pool of features in a short period of time for both classification and regression tasks.

### AutoML Tools

Automated Machine Learning (AutoML) is an emerging field in which the process of building machine learning models to model data is automated.

## References

[How to Select an ML Model?](https://www.kdnuggets.com/2021/08/select-initial-model-data-science-problem.html)

[Gettting Started with Machine Learning](https://machinelearningmastery.com/start-here/)

[Regression for Classification](https://towardsdatascience.com/regression-for-classification-hands-on-experience-8754a909a298)

[Multinomial Logistic Regression With Python](https://machinelearningmastery.com/multinomial-logistic-regression-with-python/)

[Multinomial Logistic Regression In a Nutshell](https://medium.com/ds3ucsd/multinomial-logistic-regression-in-a-nutshell-53c94b30448f)

[One-vs-Rest and One-vs-One for Multi-Class Classification](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)


## Textbooks

[Math Refresher for Scientists and Engineers](http://itc-books.weebly.com/uploads/1/4/1/0/14101304/ebooksclub.org__math_refresher_for_scientists_and_engineers.pdf)

[Trigonometry Handbook](http://www.mathguy.us/Handbooks/TrigonometryHandbook.pdf)

[Guide to the Software Engineering Body of Knowledge](https://www.computer.org/education/bodies-of-knowledge/software-engineering/v3)

[Cheatsheets](https://github.com/Neklaustares-tPtwP/Resources/tree/main/Cheat%20Sheets)


[1] M. P. Deisenroth, A. A. Faisal, and C. S. Ong, Mathematics for Machine Learning, Cambridge, UK: Cambridge University Press, ISBN: 978-1-108-47004-9, 2020.

[2] S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 3rd ed. Upper Saddle River, NJ, USA: Prentice Hall, ISBN: 978-0-13-604259-4, 2010.

[3] P. Bourque and R. E. Fairley, Guide to the Software Engineering Body of Knowledge, v. 3, IEEE, 2014. 


