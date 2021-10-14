# Frequently Asked Questions

## Do I need a Master’s Degree?

If you are going to spend the time to study AI/ML then you might as well invest in an online degree which will greatly increase your career opportunities (and a requirement for most all AI engineer positions).

[Do you need a Master’s Degree in Data Science?](https://towardsdatascience.com/do-you-need-a-masters-degree-in-data-science-you-are-asking-the-wrong-question-8c83dec8bf1b?source=rss----7f60cf5620c9---4)


## Recommended Tutorials and Books

See [Tutorials and Books](./how_to_learn.md)


## How to ask an AI/ML question?

It is usually best to share files via DM or create a thread so that other users do not have to search the channels for your files. Keep in mind that Discord channel content is unstructured, so it is difficult to search channels for individual posts.

Your post(s) should contain the following:

1. Describe the problem. 

In a few sentences describe the problem including the type of ML problem if known (classification, regression, NLP, CV, etc.).

What is the goal? classify, predict, detect, translate, etc. 

2. Describe the dataset in detail and be willing to share your dataset. 

Describe the dataset including the input features and target feature(s). 

It is best to share summary statistics of the data including counts of any discrete or categorical features including the target feature. 

It is best to share the entire dataset (if you want someone to help you then you must be open and honest).

Please note that Discord users are more than willing to donate their time to give free consulting advice but it is unethical to try to ask vague questions in an effort to get free advice on a commercial or research project that you are getting paid to do. If this is the case, you should be diligent in stating this fact up front repeatedly (do not expect other Discord users to go data mining for your original post). 

3. Avoid asking users to help debug your code. 

In general, the problem is usually not the algorithm implementation but the data preparation and feature engineering of your dataset. 

If you find yourself mired in debugging code then this should be a red flag that you need to refactor or more likely choose a simpler model (Occam's Razor). 

See [How to Choose an ML Model?](./getting_started.md)

4. Plots and graphs

It is best not to plot more than one metric on a graph since libraries such as matplotlib will automatically adjust the axes to better show the difference in values (for examples see [How to Diagnose Overfitting and Underfitting](./ml/diagnose_overfitting.md)).

The best practice (and approach used by most ML tools) is to compute several performance metrics (see [Machine Learning Error Metrics](./ml/error_metrics.md)) rather than plotting graphs. The only graph that is commonly used in ML is a plot of train/val loss to spot-check convergence of training of a model. 

Since AI/ML models are dynamic/stochastic in nature, you will get slightly different result each time you train and fit your model. Therefore, the best practice is to run the entire process (train and fit the model then evaluate the model by computing the performance metrics) many times (say 10 or 100) and save the results. Finally, compute summary statistics on the results such as mean and standard deviation. 


## How to choose a performance metric?

See [Machine Learning Error Metrics](./ml/error_metrics.md) 


## How to Choose an ML Algorithm?

See [How to Choose an ML Model?](./getting_started.md) and [Understand Machine Learning Algorithms](./ml/error_metrics.md)


## Should I start learning ML by coding an algorithm from scratch?

[How to Learn Machine Learning](https://link.medium.com/kgpRXAT9lkb)

As my professor has said "any competent software engineer can implement any AI/ML algorithm". 

It is important to know what algorithms are available for a given problem, how they work, and how to get the most out of them. However, this does not mean you need to hand-code the algorithms from scratch.

There are far too many ML algorithms for a single software developer to ever code and understand all the details needed to properly implement them (and more algorithms are constantly being developed). Just concentrating on the “more interesting” or latest algorithm is a recipe for disaster and a common beginner mistake.

In fact, there are many other concepts that are much more important than knowing how to implement an ML algorithm:

- How to properly define an ML problem?
- How to select a dataset?
- How to perform data preparation?
- How to design an ML solution?
- How to properly train an ML model?
- How to choose a Performance/Error Metric?
- How to deploy an ML solution?
- How to debug ML models?

In short, it is best to understand the big picture (top-down approach) of the ML engineering process before delving into the implementation details (bottom-up approach). 


## How to share your work?

Your are always welcome to share your work in the following Discord AI forum channels of the SHARE KNOWLEDGE Section: #share-your-projects, #share-datasets, and #our-accomplishments. 

