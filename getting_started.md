# Getting Started

I highly recommend that you refer to more than one resource (other than acikit-learn, tensorflow, and PyTorch documentation) when learning ML. 

If you have an .edu email account you can get free access to [oreilly.com](https://www.oreilly.com/) which has two good books for beginners that are available “Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow” and “Python Machine Learning: Machine Learning and Deep Learning with Python, scikit-learn, and TensorFlow”. There also many other good books on various AI/ML topics.

**NOTE:** The Medium and TowardsDataSciene articles can be viewed in a browser Private tab if you do not have a subscription. 

Here are some undergraduate topics/courses that you should know before learning AI/ML (see Math and References sections):

- Advanced Linear Algebra
- Discrete Mathematics
- Probability and Statitics
- Statistical Programming
- Computer Networks
- Computer Organization
- Operating Systems
- Distributed Computing


## Math

[Learn Data Science For Free](https://medium.com/artificialis/learn-data-science-for-free-fad0aa268c1e)

[Math Refresher for Scientists and Engineers](http://itc-books.weebly.com/uploads/1/4/1/0/14101304/ebooksclub.org__math_refresher_for_scientists_and_engineers.pdf)

[Trigonometry Handbook](http://www.mathguy.us/Handbooks/TrigonometryHandbook.pdf)

[An Introduction To Recurrent Neural Networks And The Math That Powers Them](https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/)



## Project Definition

- In a few sentences describe the problem including the type of ML problem if known (classification, regression, NLP, CV, etc.). 

- What is the goal? classify, predict, detect, translate, etc. 

- Describe the dataset including the input features and target feature(s). 

- It is best to share summary statistics of the data including counts of any discrete or categorical features including the target feature. 

- It is also best to share the entire dataset (if you want someone to help you then you must be open and honest).


## The Machine Learning Process

J. Brownlee defines a 5-step process for solving ML prediction problems called the [Applied Machine Learning Process](https://machinelearningmastery.com/start-here/#process) that is applicable to most any ML problem. 

Step 1: Define your problem
Step 2: Prepare your data
Step 3: Spot-check algorithms
Step 4: Improve results
Step 5: Present results

## How to Choose an ML Model?

[How to Select an ML Model?](https://www.kdnuggets.com/2021/08/select-initial-model-data-science-problem.html)

[Applied Machine Learning Checklist](./checklist/applied_ml_checklist.md)

Every new ML engineer finds that they need to decide what model to use for a problem.

There are many models to choose from with seemingly endless variants, but there are usually only slight alterations needed to change a regression model into a classification model and vice versa.

There are a lot of models to choose from, so consider starting with regression models to obtain a simple baseline for comparison. The work has already been done for you with the standard scikit-learn supervised learning packages, so you only need to select what option you want.

The first step in solving an ML problem is to try a simple algorithm (such as Linear or Logistic Regression) as a baseline model which is used later to evaluate your model choice(s) which should perform better than the baseline. 

The best practice is to evaluate many algorithms (say 10-20) using an [AutoML Tool](./tips/automl_tools.md) and [ML Tool](./tips/ml_tools.md) such as Orange or PyCaret that c such as Orange or PyCaret then narrow the choices to a few models based on accuracy and error metrics. Then, create a test harness to fully explore the candidates.

In general, you should have evaluated 10-20 models before trying to evaluate more complex models such as neural networks (a common beginner mistake).

Keep in mind that an accuracy of 50% is equivalent to random guessing (flip of a coin). Thus, your models should have an accuracy of at least 70-80% or better before optimization/tuning of hyperparameters. Otherwise, this should be a red flag that you need to select a different model and/or spend more time on data preparation and feature engineering. 

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


----------


## Understand Machine Learning Algorithms

You need to know what algorithms are available for a given problem, how they work, and how to get the most out of them. However, this does not mean you need to hand-code the algorithms from scratch.

In fact, there are many other concepts that are much more important when implmenting ML models:

- How to define an ML problem?
- How to select a dataset?
- Data Preparation Techniques
- Feature Engineering Techniques
- How to design an ML solution?
- How to properly train an ML model?

- How to choose a Performance/Error Metric?

- How to display ML results?

- How to deploy an ML solution?
- How to debug ML models?

- Parametric vs Nonparametric Algorithms
- Supervised vs Unsupervised Algorithms
- The Bias-Variance Trade-Off
- How to Diagnose/Fix Overfitting and Underfitting?

- How to create a data pipeline?

- How to deal with small datasets?
- How to deal with imbalanced datasets?

- Anomaly Detection
- AutoML

Even if you are an experienced data scientist, you should still know the performance of simpler models on your dataset/problem since they are easy to implement and test. 

In fact, it is considered a best practice to run test datasets to see if your chosen machine learning model outperforms a recognized benchmark.


## Feature Engineering Tools

Feature engineering (FE) techniques for ML are a fundamental ML topic but one that is often overlooked or deceptively simple.

There are many tools that will help you to automate the entire FE process and produce a large pool of features in a short period of time for both classification and regression tasks.


## AutoML Tools

Automated Machine Learning (AutoML) is an emerging field in which the process of building machine learning models to model data is automated.

There are a plethora of [AutoML Tools](./tips/automl_tools.md) and [ML Tools](./tips/ml_tools.md) such as Orange or PyCaret that can be used to easily and quickly evaluate many models on a dataset.


## References

[Gettting Started with Machine Learning](https://machinelearningmastery.com/start-here/)

[One-vs-Rest and One-vs-One for Multi-Class Classification](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)


## Python

[Introduction to APIs in Python](https://towardsdatascience.com/quick-fire-guide-to-apis-in-python-891dd98c8877?source=rss----7f60cf5620c9---4)


## More References

[Guide to the Software Engineering Body of Knowledge](https://www.computer.org/education/bodies-of-knowledge/software-engineering/v3)

[Cheatsheets](https://github.com/Neklaustares-tPtwP/Resources/tree/main/Cheat%20Sheets)


[1] M. P. Deisenroth, A. A. Faisal, and C. S. Ong, Mathematics for Machine Learning, Cambridge, UK: Cambridge University Press, ISBN: 978-1-108-47004-9, 2020.

[2] S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 3rd ed. Upper Saddle River, NJ: Prentice Hall, ISBN: 978-0-13-604259-4, 2010.

[3] P. Bourque and R. E. Fairley, Guide to the Software Engineering Body of Knowledge (SWEBOK) v. 3, IEEE, 2014. 

[4] B. Siciliano, L. Sciavicco, L. Villani, and G. Oriolo, Robotics: Modelling, Planning and Control, London: Springer, ISBN 978-1-84628-641-4, 2010. 


W. McKinney, Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython, 2nd ed., O’Reilly Media, ISBN: 978-1491957660, 2017. 

E. Alpaydin, Introduction to Machine Learning, 3rd ed., MIT Press, ISBN: 978-0262028189, 2014. 

S. Raschka. and V. Mirjalili, Python Machine Learning, 2nd ed. Packt, ISBN: 978-1787125933, 2017. 

S. Bird, E. Klein, and E. Loper. Natural Language Processing with Python – Analyzing Text with the Natural Language Toolkit. Available online: https://www.nltk.org/book/.

R. H. Arpaci-Dusseau and A. C. Arpaci-Dusseau, Operating Systems: Three Easy Pieces, 2018, v. 1.01, Available online: https://pages.cs.wisc.edu/~remzi/OSTEP/





