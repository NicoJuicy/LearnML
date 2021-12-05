# Getting Started

<!-- MarkdownTOC -->

- Math
- AI vs ML vs DL
- Project Definition
- The Machine Learning Process
- How to Choose an ML Model?
    - Model Selection Criteria
    - Why Simple Models?
    - Save yourself some headache
- Multinomial Logistic Regression \(MLR\)
- Understand Machine Learning Algorithms
- Feature Engineering Tools
- AutoML Tools
- References

<!-- /MarkdownTOC -->


I highly recommend that you refer to more than one resource (other than acikit-learn, tensorflow, and PyTorch documentation) when learning ML. 

If you have an .edu email account you can get free access to [oreilly.com](https://www.oreilly.com/) which has  good books on a variety of AI/ML topics.

**NOTE:** The Medium and TowardsDataSciene articles can be viewed in a browser Private tab if you do not have a subscription. 

Here are some undergraduate topics/courses that you should know before learning AI/ML (see Math and References sections):

- Calculus
- Discrete Mathematics
- Linear Algebra
- Probability and Statistics
- Statistical Programming

The following courses are usually required for a computer science degree:

- Computer Networks
- Computer Organization
- Operating Systems
- Distributed Computing


[What Google Recommends You do Before Taking Their Machine Learning or Data Science Course](https://www.kdnuggets.com/2021/10/google-recommends-before-machine-learning-data-science-course.html)


## Math

[Math Refresher for Scientists and Engineers](http://itc-books.weebly.com/uploads/1/4/1/0/14101304/ebooksclub.org__math_refresher_for_scientists_and_engineers.pdf)

[Trigonometry Handbook](http://www.mathguy.us/Handbooks/TrigonometryHandbook.pdf)

[Lessons on How to Lie with Statistics](https://towardsdatascience.com/lessons-from-how-to-lie-with-statistics-57060c0d2f19?gi=27adc9e8b44a)

[An Introduction To Recurrent Neural Networks And The Math That Powers Them](https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/)

[Learn Data Science For Free](https://medium.com/artificialis/learn-data-science-for-free-fad0aa268c1e)


## AI vs ML vs DL

[Artificial intelligence](https://en.wikipedia.org/wiki/Artificial_intelligence?wprov=sfti1) (AI) is the study of ways to build intelligent programs and machines that can creatively solve problems which has always been considered a human prerogative.

[Machine learning](https://en.wikipedia.org/wiki/Machine_learning?wprov=sfti1) is a subset of AI that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. In ML, there are different algorithms (e.g. neural networks) that help to solve problems.

[Deep learning](https://en.wikipedia.org/wiki/Deep_learning?wprov=sfti1) (DL) is a subset of machine learning which uses neural networks that have a structure similar to the human neural system.

<img width=600 alt="Diagram showing comparison" src="https://serokell.io/files/zx/zxwju3ha.Machine-learning-vs-deep-learning.jpg" />

[Overview of AI and ML](https://ocw.mit.edu/resources/res-ll-005-mathematics-of-big-data-and-machine-learning-january-iap-2020/lecture-notes/MITRES_LL_005IAP20_Supplemental_Ses01.pdf)


## Project Definition

In a few sentences describe the problem including the type of ML problem if known (classification, regression, NLP, CV, etc.). 

What is the goal? classify, predict, detect, translate, etc. 

Describe the dataset including the input features and target feature(s). 

It is best to share summary statistics of the data including counts of any discrete or categorical features including the target feature. 

It is best to share the entire dataset (if you want someone to help you then you must be open and honest).


## The Machine Learning Process

J. Brownlee defines a 5-step process for solving ML prediction problems called the [Applied Machine Learning Process](https://machinelearningmastery.com/start-here/#process) that is applicable to most any ML problem. 

Step 1: Define your problem
Step 2: Prepare your data
Step 3: Spot-check algorithms
Step 4: Improve results
Step 5: Present results


## How to Choose an ML Model?

Every new ML engineer finds that they need to decide what model to use for a problem.

There are many models to choose from with seemingly endless variants, but there are usually only slight alterations needed to change a regression model into a classification model and vice versa.

There are a lot of models to choose from, so consider starting with regression models to obtain a simple baseline for comparison. The work has already been done for you with the standard scikit-learn supervised learning packages, so you only need to select what option you want.

The first step in solving an ML problem is to try a simple algorithm (such as Linear or Logistic Regression) as a baseline model which is used later to evaluate your model choice(s) which should perform better than the baseline. 

Next, the best practice is to evaluate many algorithms (say 10-20) using an [AutoML Tool](./tips/automl_tools.md) or [ML Tool](./tips/ml_tools.md) such as Orange and PyCaret then narrow the choices to a few models based on accuracy and error metrics. Then, create a test harness to fully explore the candidates.

In general, you should have evaluated 10-20 models before trying to evaluate more complex models such as neural networks.

Keep in mind that an accuracy of 50% is equivalent to random guessing (flip of a coin). Thus, your models should have an accuracy of at least 70-80% or better before optimization/tuning of hyperparameters. Otherwise, this should be a red flag that you need to select a different model and/or spend more time on data preparation and feature engineering. 

### Model Selection Criteria

Given the following seven criteria it will help to shortlist your choices to be able to apply them in a short time.

**1. Explainability**

There is a trade-off between explainability and model performance. 

Using a more complex model will often increase the performance but it will be more difficult to interpret. 

If there is no need to explain the model and its output to a non-technical audience, then more complex models could be used such as ensemble learners and deep neural networks.

**2. In memory vs out memory**

It is important to consider the size of your data and RAM of the server or your personal computer that training will occur on. 

If the RAM can handle all of the training data then you can choose from a wide variety of machine learning algorithms. 

If the RAM cannot handle the training data, therefore different incremental learning algorithms which can improve the model by adding more training data gradually would be a good choice.

**3. Number of features and examples**

The number of training examples and the number of features per example is also important in model selection. 

If you have a small number of examples and features then a simple learner would be a great choice such as a decision tree and KNN. 

If you have a small number of examples and a large number of features, SVM and gaussian processes would be a good choice as they can handle a large number of features, but are very modest in their capacity. 

If you have a large number of examples then deep neural networks and boosting algorithms would be a great choice since they can handle millions of examples and features. 

**4. Categorical vs numerical features**

The type of features is also an important when choosing a model. 

Some machine learning algorithms cannot handle categorical features such as linear regressions and you have to convert them into numerical features while other algorithms can handle categorical features and numerical features such as decision trees and random forests.

**5. Normality of data**

If your data is linearly separable or can be modeled using a linear model then SVM with linear kernel or logistic regression or linear regression model could be used. 

If your data is non-linearly separable or non-linearly modeled then deep neural networks or ensemble learners would be a good choice.

**6. Training speed**

The available time for training is also another important criterion to choose your training model on. 

Simple algorithms such as logistic and linear regression or decision trees can be trained in a short time. 

Complex algorithms such as neural networks and ensemble learners are known to be slow to train. 

If you have access to a multi-core machine this could significantly reduce the training time of more complex algorithms.

**7. Prediction speed**

The speed of generating the results is a another important criterion to choose a model. 

If your model will be used in real-time or in a production environment, it should be able to generate the results with very low latency. 

Algorithms such as SVMs, linear and logistic regression, and some types of neural networks are extremely fast at the prediction time. 

You should also consider where uiu premises you will deploy your model on. If you are using the models for  analysis or theoretical purposes, your prediction time can be longer which means you could use ensemble algorithms and very deep neural networks.


### Why Simple Models?

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


----------


## Understand Machine Learning Algorithms

You need to know what algorithms are available for a given problem, how they work, and how to get the most out of them. However, this does not mean you need to hand-code the algorithms from scratch.

Even if you are an experienced ML engineer, you should still know the performance of simpler models on your dataset/problem since they are easy to implement and test. 

In fact, there are many other concepts that are much more important when implmenting ML models:

- How to define an ML problem?
- How to select a dataset?
- How to perform data preparation?
- How to perform feature engineering?
- How to select an ML algorithm?
- How to choose a performance metric?
- How to train an ML model?
- How to present ML results?
- How to design and deploy an ML solution?
- How to debug ML models?


[TODO: Additional topics below]

- Parametric vs Nonparametric Algorithms
- Supervised vs Unsupervised Algorithms
- The Bias-Variance Trade-Off
- How to Diagnose/Fix Overfitting and Underfitting?

- How to create a data pipeline?

- How to deal with small datasets?
- How to deal with imbalanced datasets?

- Anomaly Detection
- AutoML



## Feature Engineering Tools

Feature engineering (FE) techniques for ML are a fundamental ML topic but one that is often overlooked or deceptively simple.

There are many tools that will help you to automate the entire FE process and produce a large pool of features in a short period of time for both classification and regression tasks.


## AutoML Tools

Automated Machine Learning (AutoML) is an emerging field in which the process of building machine learning models to model data is automated.

[A Beginner’s Guide to End to End Machine Learning](https://link.medium.com/ao2ANbsVDlb)

There are a plethora of [AutoML Tools](./tips/automl_tools.md) and [ML Tools](./tips/ml_tools.md) such as Orange or PyCaret that can be used to easily and quickly evaluate many models on a dataset.


## References

[Gettting Started with Machine Learning](https://machinelearningmastery.com/start-here/)

[How to Select an ML Model?](https://www.kdnuggets.com/2021/08/select-initial-model-data-science-problem.html)

[Brief Guide for Machine Learning Model Selection](https://medium.com/mlearning-ai/brief-guide-for-machine-learning-model-selection-a19a82f8bdcd)

[Applied Machine Learning Checklist](./checklist/applied_ml_checklist.md)

[Regression for Classification | Hands on Experience](https://towardsdatascience.com/regression-for-classification-hands-on-experience-8754a909a298)

[A Practical Guide to Linear Regression](https://towardsdatascience.com/a-practical-guide-to-linear-regression-3b1cb9e501a6)


[End-to-end machine learning project: Telco customer churn](https://towardsdatascience.com/end-to-end-machine-learning-project-telco-customer-churn-90744a8df97d?source=rss----7f60cf5620c9---4)

[How to start contributing to open-source projects](https://towardsdatascience.com/how-to-start-contributing-to-open-source-projects-41fcfb654b2e)


