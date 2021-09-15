# Machine Learning Algorithms

[Understand Machine Learning Algorithms](https://machinelearningmastery.com/start-here/#algorithms)

Machine learning is about machine learning algorithms.

You need to know what algorithms are available for a given problem, how they work, and how to get the most out of them.

Here is how to get started with machine learning algorithms:

**Step 1:** Discover the different types of machine learning algorithms.

**Step 2:** Discover the foundations of machine learning algorithms.

- How Machine Learning Algorithms Work
- Parametric and Nonparametric Algorithms
- Supervised and Unsupervised Algorithms
- The Bias-Variance Trade-Off
- Overfitting and Underfitting With Algorithms

**Step 3:** Discover how top machine learning algorithms work.

- Machine Learning Algorithms Mini-Course
- Master Machine Learning Algorithms (my book)



----------


# Categories of ML Algorithms

[A Tour of Machine Learning Algorithms](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)

There are two common approaches of categorizing ML algorithms which you will come across.

- Grouping of algorithms by learning style

- Grouping of algorithms by similarity in form or function


## Algorithms Grouped by Learning Style

There are different ways an algorithm can model a problem based on its interaction with the experience, environment, or whatever we want to call the input data.

It is popular in machine learning and AI textbooks to consider the learning styles that an algorithm can adopt.

This _taxonomy_ or way of organizing machine learning algorithms is useful because it forces you to think about the roles of the input data and model preparation process so that you can select one that is the most appropriate for your problem.


### Supervised Learning

In supervised learning, the input data is called training data and has a known label or result. 

The model is prepared through a training process in which it is required to make predictions and is corrected when those predictions are wrong. The training process continues until the model achieves a desired level of accuracy on the training data.

Examples problems: classification and regression.

Example algorithms: Logistic Regression and Back-Propagation Neural Network.

### Unsupervised Learning

In unsupervised learning, the input data is not labeled, so the data does not have a known result.

The model is prepared by deducing structures present in the input data which may be to extract general rules through a mathematical process to systematically reduce redundancy or to organize data by similarity.

Example problems: clustering, dimensionality reduction and association rule learning.

Example algorithms: the Apriori algorithm and K-Means.

### Semi-Supervised Learning

In semi-supervised learning, the input data is a mixture of labeled and unlabelled examples.

There is a desired prediction problem but the model must learn the structures to organize the data as well as make predictions.

Example problems: classification and regression.

Example algorithms are extensions to other flexible methods that make assumptions about how to model the unlabeled data.


## Overview of Machine Learning Algorithms

When crunching data to model business decisions, you are most usually using supervised and unsupervised learning methods.

A hot topic at the moment is semi-supervised learning methods in areas such as image classification where there are large datasets with very few labeled examples.


## Algorithms Grouped By Similarity

Algorithms are often grouped by similarity in terms of their function (how they work). For example, tree-based methods and neural network inspired methods.

This is perhaps the most useful way to group algorithms and it is the approach we will use in [1].

This is a useful grouping method but it is not perfect. However, there are algorithms that could just as easily fit into multiple categories.

### Regression Algorithms

**Regression** is concerned with modeling the relationship between variables that is iteratively refined using a measure of error in the predictions made by the model.

Regression methods are a workhorse of statistics and have been applied to statistical machine learning. 

The most common regression algorithms are:

- Ordinary Least Squares Regression (OLSR)
- Linear Regression
- Logistic Regression
- Stepwise Regression
- Multivariate Adaptive Regression Splines (MARS)
- Locally Estimated Scatterplot Smoothing (LOESS)

The primary assumptions of linear regression are:

1. **Linearity:** There is a linear relationship between the outcome and predictor variable(s).

2. **Normality:** The residuals or errors follow a normal distribution. The error is calculated by subtracting the predicted value from the actual value.

3. **Homoscedasticity:** The variability in the dependent variable is equal for all values of the independent variable(s).

### Instance-based Algorithms

Instance-based learning model is a decision problem with instances or examples of training data that are deemed important or required to the model.

Such methods typically build up a database of example data and compare new data to the database using a similarity measure in order to find the best match and make a prediction. 

For this reason, instance-based methods are also called winner-take-all methods and memory-based learning. 

The focus is on the representation of the stored instances and similarity measures used between instances.

The most popular instance-based algorithms are:

- k-Nearest Neighbor (kNN)
- Learning Vector Quantization (LVQ)
- Self-Organizing Map (SOM)
- Locally Weighted Learning (LWL)
- Support Vector Machines (SVM)

### Decision Tree Algorithms

Decision tree methods construct a model of decisions made based on actual values of attributes in the data.

Decisions fork in tree structures until a prediction decision is made for a given record. 

Decision trees are trained on data for classification and regression problems. 

Decision trees are often fast and accurate and a big favorite in machine learning.

The most popular decision tree algorithms are:

- Classification and Regression Tree (CART)
- Iterative Dichotomiser 3 (ID3)
- C4.5 and C5.0 (different versions of a powerful approach)
- Chi-squared Automatic Interaction Detection (CHAID)
- Decision Stump
- M5
- Conditional Decision Trees


### Bayesian Algorithms

Bayesian methods are those that explicitly apply Bayes’ Theorem for problems such as classification and regression.

The most popular Bayesian algorithms are:

- Naive Bayes
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Averaged One-Dependence Estimators (AODE)
- Bayesian Belief Network (BBN)
- Bayesian Network (BN)


### Clustering Algorithms

Clustering (like regression) describes the class of problem and the class of methods.

Clustering methods are typically organized by the modeling approaches such as centroid-based and hierarchal. 

All clustering methods are concerned with using the inherent structures in the data to best organize the data into groups of maximum commonality.

The most popular clustering algorithms are:

- k-Means
- k-Medians
- Expectation Maximization (EM)
- Hierarchical Clustering


### Association Rule Learning Algorithms

Association rule learning methods extract rules that best explain observed relationships between variables in data.

These rules can discover important and commercially useful associations in large multidimensional datasets that can be exploited by an organization.

The most popular association rule learning algorithms are:

- Apriori algorithm
- Eclat algorithm


### Artificial Neural Network Algorithms

Artificial Neural Networks are models that are inspired by the structure and/or function of biological neural networks.

They are a class of _pattern matching_ that are commonly used for regression and classification problems but are really an enormous subfield comprised of hundreds of algorithms and variations for all manner of problem types.

The most popular artificial neural network algorithms are:

- Perceptron
- Multilayer Perceptrons (MLP)
- Back-Propagation
- Stochastic Gradient Descent
- Hopfield Network
- Radial Basis Function Network (RBFN)

### Deep Learning Algorithms

Deep Learning methods are a modern update to Artificial Neural Networks that exploit abundant cheap computation.

They are concerned with building much larger and more complex neural networks and many methods are concerned with very large datasets of labelled analog data such as image, text. audio, and video.

The most popular deep learning algorithms are:

- Convolutional Neural Network (CNN)
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory Networks (LSTMs)
- Stacked Auto-Encoders
- Deep Boltzmann Machine (DBM)
- Deep Belief Networks (DBN)


### Dimensionality Reduction Algorithms

Like clustering methods, dimensionality reduction seek and exploit the inherent structure in the data, but in an unsupervised manner or order to summarize or describe data using less information.

This can be useful to visualize dimensional data or to simplify data which can then be used in a supervised learning method. 

Many of these methods can be adapted for use in classification and regression.

- Principal Component Analysis (PCA)
- Principal Component Regression (PCR)
- Partial Least Squares Regression (PLSR)
- Sammon Mapping
- Multidimensional Scaling (MDS)
- Projection Pursuit
- Linear Discriminant Analysis (LDA)
- Mixture Discriminant Analysis (MDA)
- Quadratic Discriminant Analysis (QDA)
- Flexible Discriminant Analysis (FDA)


### Ensemble Algorithms

Ensemble methods are models composed of multiple weaker models that are independently trained and whose predictions are combined in some way to make the overall prediction.

Much effort is put into what types of weak learners to combine and the ways in which to combine them. 

Ensemble is a very powerful class of techniques and as such is very popular.

- Boosting
- Bootstrapped Aggregation (Bagging)
- AdaBoost
- Weighted Average (Blending)
- Stacked Generalization (Stacking)
- Gradient Boosting Machines (GBM)
- Gradient Boosted Regression Trees (GBRT)
- Random Forest


Other Lists of Machine Learning Algorithms

How to Study Machine Learning Algorithms

How to Run Machine Learning Algorithms


### Decision Trees

[Classification And Regression Trees for Machine Learning](https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/)

Decision Trees are an important type of algorithm for predictive modeling machine learning.

The classical decision tree algorithms have been around for decades and modern variations like random forest are among the most powerful techniques available.

[How to Tune the Number and Size of Decision Trees with XGBoost in Python](https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/)

Gradient boosting involves the creation and addition of decision trees sequentially, each attempting to correct the mistakes of the learners that came before it.

This raises the question as to how many trees (weak learners or estimators) to configure in your gradient boosting model and how big each tree should be



## References

[1] [A Tour of Machine Learning Algorithms](https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/)


—————————-



## Examples of Algorithm Lists To Create

Below are 10 examples of machine learning algorithm lists that you could create.

- Regression algorithms
- SVM algorithms
- Data projection algorithms
- Deep learning algorithms
- Time series forecasting algorithms
- Rating system algorithms
- Recommender system algorithms
- Feature selection algorithms
- Class imbalance algorithms
- Decision tree algorithms


## Tips for Creating Algorithm List

- Start with why you want the list and use that to define the type of list to create.

- Only capture the algorithm properties you actually need, keep it as simple as possible.

