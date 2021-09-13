# Imbalanced Classification

[A Gentle Introduction to Imbalanced Classification](https://machinelearningmastery.com/what-is-imbalanced-classification/) 

[Standard Machine Learning Datasets for Imbalanced Classification](https://machinelearningmastery.com/standard-machine-learning-datasets-for-imbalanced-classification/)


An imbalanced classification problem is a problem that involves predicting a class label where the distribution of class labels in the training dataset is skewed (there are many more examples for one class than another class). 

Many real-world classification problems have an imbalanced class distribution, therefore it is important for machine learning practitioners to get familiar with working with these types of problems.


Imbalanced classifications pose a challenge for predictive modeling sincr most of the machine learning algorithms used for classification were designed around the assumption of an equal number of examples for each class which results in models that have poor predictive performance, specifically for the minority class. 

This is a problem because the minority class is usually more important and therefore the problem is more sensitive to classification errors for the minority class than the majority class.

These types of problems often require the use of specialized performance metrics and learning algorithms as the standard metrics and methods are unreliable or fail completely.

Given measurements of a flower (observation), we may predict the likelihood (probability) of the flower being an example of each of twenty different species of flower.

The number of classes for a predictive modeling problem is typically fixed when the problem is framed or described and the number of classes usually does not change.

A classification predictive modeling problem may have two class labels called binary classification or the problem may have more than two classes such as three, 10, or even hundreds of classes called multi-class classification problems.

- Binary Classification Problem: A classification predictive modeling problem where all examples belong to one of two classes.

- Multiclass Classification Problem: A classification predictive modeling problem where all examples belong to one of three classes.

A training dataset is a number of examples from the domain that include both the input data (measurements) and the output data (class label).

Depending on the complexity of the problem and the types of models we may choose to use, we may need tens, hundreds, thousands, or even millions of examples from the domain to constitute a training dataset.

The training dataset is used to better understand the input data to help best prepare it for modeling:

-  to evaluate a suite of different modeling algorithms

- to tune the hyperparameters of a chosen model. 

- to train a final model on all available data that we can use to make predictions for new samples from the problem domain.

## Imbalanced Classification Problems

The number of examples that belong to each class may be referred to as the _class distribution_.

Terminology: Unbalance refers to a class distribution that was balanced and is now no longer balanced, whereas imbalanced refers to a class distribution that is inherently not balanced.

There are other less general names that may be used to describe these types of classification problems, such as:

- Rare event prediction
- Extreme event prediction
- Severe class imbalance

It is common to describe the imbalance of classes in a dataset in terms of a _ratio_. 

For example, an imbalanced binary classification problem with an imbalance of 1 to 100 (1:100) means that for every one example in one class, there are 100 examples in the other class.

Another way to describe the imbalance of classes in a dataset is to summarize the class distribution as percentages of the training dataset. 

For example, an imbalanced multiclass classification problem may have 80 percent examples in the first class, 18 percent in the second class, and 2 percent in a third class.

## Causes of Class Imbalance

The imbalance to the class distribution in an imbalanced classification predictive modeling problem may have many causes.

There are two main groups of causes for the imbalance we may want to consider: data sampling and properties of the domain.

It is possible that the imbalance in the examples across the classes is caused by the way the examples were collected or sampled from the problem domain which might involve biases introduced during data collection and errors made during data collection.

- Biased Sampling
- Measurement Error

The imbalance might be a property of the problem domain.

For example, the natural occurrence or presence of one class may dominate other classes which may be because the process that generates observations in one class is more expensive in time, cost, computation, or other resources. 

Thus, it is often infeasible or intractable to simply collect more samples from the domain in order to improve the class distribution. Instead, a model is required to learn the difference between the classes.

## Challenge of Imbalanced Classification

The imbalance of the class distribution will vary across problems.

A classification problem may be a little skewed such as if there is a slight imbalance. 

Alternately, the classification problem may have a severe imbalance where there might be hundreds or thousands of examples in one class and tens of examples in another class for a given training dataset.

- Slight Imbalance. An imbalanced classification problem where the distribution of examples is uneven by a small amount in the training dataset (4:6).

- Severe Imbalance. An imbalanced classification problem where the distribution of examples is uneven by a large amount in the training dataset (1:100 or more).

Most of the contemporary works in class imbalance concentrate on imbalance ratios ranging from 1:4 up to 1:100. 

In real-life applications such as fraud detection or cheminformatics, we may deal with problems with imbalance ratio ranging from 1:1000 up to 1:5000.

A slight imbalance is often not a concern and the problem can often be treated like a normal classification predictive modeling problem. 

A severe imbalance of the classes can be challenging to model and may require the use of specialized techniques.

The class or classes with abundant examples are called the **major** or **majority classes** whereas the class with few examples (and there is typically just one) is called the **minor** or **minority class**.

- Majority Class: The class (or classes) in an imbalanced classification predictive modeling problem that has many examples.

- Minority Class: The class in an imbalanced classification predictive modeling problem that has few examples.

The abundance of examples from the majority class (or classes) can swamp the minority class. 

Most machine learning algorithms for classification predictive models are designed and demonstrated on problems that assume an equal distribution of classes which means that a naive application of a model may focus on learning the characteristics of the abundant observations only while neglecting the examples from the minority class. 

Imbalanced classification remains an open problem generally and practically must be identified and addressed specifically for each training dataset.

## Examples of Imbalanced Classification

Many of the classification predictive modeling problems that we are interested in solving in practice are imbalanced.

Notice that most of the examples are likely binary classification problems and the examples from the minority class are rare, extreme, abnormal, or unusual in some way.

Also notice that many of the domains are described as “detection,” highlighting the desire to discover the minority class amongst the abundant examples of the majority class.


## References

[Step-By-Step Framework for Imbalanced Classification Projects](https://machinelearningmastery.com/framework-for-imbalanced-classification-projects/)

[Fitting Linear Regression Models on Counts Based Data](https://towardsdatascience.com/fitting-linear-regression-models-on-counts-based-data-ba1f6c11b6e1)

[Discrete Probability Distributions for Machine Learning](https://machinelearningmastery.com/discrete-probability-distributions-for-machine-learning/)

[SMOTE for Imbalanced Classification with Python](https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/)

[Multi-Class Imbalanced Classification](https://machinelearningmastery.com/multi-class-imbalanced-classification/)

[How to handle Multiclass Imbalanced Data? Not SMOTE](https://towardsdatascience.com/how-to-handle-multiclass-imbalanced-data-say-no-to-smote-e9a7f393c310)



