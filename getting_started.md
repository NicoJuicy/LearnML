# Getting Started

**NOTE:** The Medium and TowardsDataSciene articles are best viewed in a browser Private tab.

## Math

[Learn Data Science For Free](https://medium.com/artificialis/learn-data-science-for-free-fad0aa268c1e)

[Math Refresher for Scientists and Engineers](http://itc-books.weebly.com/uploads/1/4/1/0/14101304/ebooksclub.org__math_refresher_for_scientists_and_engineers.pdf)

[Trigonometry Handbook](http://www.mathguy.us/Handbooks/TrigonometryHandbook.pdf)

[An Introduction To Recurrent Neural Networks And The Math That Powers Them](https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/)


## Statistics

[What does RMSE really mean?](https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e)

[Reasoning under Uncertainty (Chapters 13 and 14.1 - 14.4)](http://pages.cs.wisc.edu/~dyer/cs540/notes/uncertainty.html)

[Conditional independence in general](http://www.cs.columbia.edu/~kathy/cs4701/documents/conditional-independence-bn.txt)


[Important Statistics Data Scientists Need to Know](https://www.kdnuggets.com/2021/09/important-statistics-data-scientists.html)

[Statistics in Python — Understanding Variance, Covariance, and Correlation](https://towardsdatascience.com/statistics-in-python-understanding-variance-covariance-and-correlation-4729b528db01?source=rss----7f60cf5620c9---4)

Here is a list of the topics covered in thr article:

- Descriptive vs. Inferential Statistics
- Data Types
- Probability and Bayes’ Theorem
- Measures of Central Tendency
- Skewness
- Kurtosis
- Measures of Dispersion
- Covariance
- Correlation
- Probability Distributions
- Hypothesis Testing
- Regression


### Correlation

The correlation between two random variables measures both the strength and direction of a linear relationship that exists between them. 

There are two ways to measure correlation:

- Pearson Correlation Coefficient: captures the strength and direction of the linear association between two continuous variables

- Spearman’s Rank Correlation Coefficient: determines the strength and direction of the monotonic relationship which exists between two ordinal (categorical) or continuous variables.

Understanding the correlations between the various columns in your dataset is an important part of the process of preparing your data for machine learning. 

You want to train your model using the columns that has the highest correlation with the target/label of your dataset.

Like covariance, the sign of the pearson correlation coefficient indicates the direction of the relationship. However, the values of the Pearson correlation coefficient is contrained to be between -1 and 1. 

Based on the value, you can deduce the following degrees of correlation:

- Perfect: values near to ±1

- High degree: values between ±0.5 and ±1

- Moderate degree: values between ±0.3 and ±0.49

- Low degree:values below ±0.29

- No correlation: values close to 0

### Which method should you use? Pearson or Spearman’s

So which method should you use? 

- Pearson correlation describes _linear_ relationships and spearman correlation describes _monotonic_ relationships. 

- A scatter plot would be helpful to visualize the data — if the distribution is linear, use Pearson correlation. If it is monotonic, use Spearman correlation.

- You can apply both the methods and check which is performing the best. 

  If the results show spearman rank correlation coefficient is greater than Pearson coefficient, it means your data has monotonic relationships and not linear (see example above).

----------


## Project Definition

- In a few sentences describe the problem including the type of ML problem if known (classification, regression, NLP, CV, etc.). 

- What is the goal? classify, predict, detect, translate, etc. 

- Describe the dataset including the input features and target feature(s). 

- It is best practice to share summary statistics of the data including counts of any discrete or categorical features including the target feature. 

- It is also best practice to share the entire dataset (if you want someone to help you then you must be open and honest).

----------


## The Machine Learning Process

J. Brownlee defines a 5-step process for solving ML prediction problems called the [Applied Machine Learning Process](https://machinelearningmastery.com/start-here/#process) that is applicable to most any ML problem. 

Step 1: Define your problem
Step 2: Prepare your data
Step 3: Spot-check algorithms
Step 4: Improve results
Step 5: Present results


## How to Select an ML Model?

[How to Select an ML Model?](https://www.kdnuggets.com/2021/08/select-initial-model-data-science-problem.html)

Every new ML engineer finds that they need to decide what model to use for a problem.

There are many models to choose from with seemingly endless variants.

There are usually only slight alterations needed to change a regression model into a classification model and vice versa.

There are a lot of models to choose from, so consider starting with regression models to obtain a simple baseline for comparison. 

The model you choose should do better than simple regression. 

This work has already been done for you with the standard python supervised learning packages, so you only need to select what option you want.

The best practice is to evaluate many algorithms (10-20) using an AutoML tool such as Orange or PyCaret then narrow the choices to a few models based on accuracy and create a test harness to fully explore the candidates. 

### Why Simple Models?

[Regression for Classification](https://towardsdatascience.com/regression-for-classification-hands-on-experience-8754a909a298)

The two most common regression algorithms are:

- Linear Regression
- Logistic Regression

Why should you start with these simple models? It is likely that your problem does not need anything fancy.

Busting out some deep learning model and spending hundreds on AWS fees to get only a slight accuracy bump is not worth it.

These two models have been studied for decades and are some of the most well-understood models in ML.

They are easily interpretable. Both models are linear, so their inputs translate to their output in a way that you could calculate by hand.

### Save yourself some headache
   
Even if you are an experienced data scientist, you should still know the performance of these models on your problem since they are so effortless to implement and test.

To convert regression problem to classificatiom problem, there are two common solutions:

- Logistic Regression: binary classification

- Softmax Regression: multiclass classification



## Multinomial Logistic Regression (MLR)

[Multinomial Logistic Regression With Python](https://machinelearningmastery.com/multinomial-logistic-regression-with-python/)

[Multinomial Logistic Regression In a Nutshell](https://medium.com/ds3ucsd/multinomial-logistic-regression-in-a-nutshell-53c94b30448f)

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
- How to choose a Performance/Error Metric?
- How to deploy an ML solution?
- How to debug ML models?

- Parametric vs Nonparametric Algorithms
- Supervised vs Unsupervised Algorithms
- The Bias-Variance Trade-Off
- How to Diagnose/Fix Overfitting and Underfitting?

- Data Preparation Techniques
- Feature Engineering Techniques
- How to create a data pipeline?

- How to deal with small datasets?
- How to deal with imbalanced datasets?

- Anomaly Detection
- AutoML

Even if you are an experienced data scientist, you should still know the performance of simpler models on your dataset/problem since they are easy to implement and test. 

In fact, it is considered a best practice to run test datasets to see if your chosen machine learning model outperforms a recognized benchmark.

### Feature Engineering Tools

Feature engineering (FE) techniques for ML are a fundamental ML topic but one that is often overlooked or deceptively simple.

There are many tools that will help you to automate the entire FE process and produce a large pool of features in a short period of time for both classification and regression tasks.

### AutoML Tools

Automated Machine Learning (AutoML) is an emerging field in which the process of building machine learning models to model data is automated.



## References

[Gettting Started with Machine Learning](https://machinelearningmastery.com/start-here/)

[One-vs-Rest and One-vs-One for Multi-Class Classification](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)


## Python

[Introduction to APIs in Python](https://towardsdatascience.com/quick-fire-guide-to-apis-in-python-891dd98c8877?source=rss----7f60cf5620c9---4)


## More References

[Guide to the Software Engineering Body of Knowledge](https://www.computer.org/education/bodies-of-knowledge/software-engineering/v3)

[Cheatsheets](https://github.com/Neklaustares-tPtwP/Resources/tree/main/Cheat%20Sheets)


[1] M. P. Deisenroth, A. A. Faisal, and C. S. Ong, Mathematics for Machine Learning, Cambridge, UK: Cambridge University Press, ISBN: 978-1-108-47004-9, 2020.

[2] S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 3rd ed. Upper Saddle River, NJ, USA: Prentice Hall, ISBN: 978-0-13-604259-4, 2010.

[3] P. Bourque and R. E. Fairley, Guide to the Software Engineering Body of Knowledge (SWEBOK) v. 3, IEEE, 2014. 


