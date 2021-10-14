# Frequently Asked Questions

## Do I need a Master’s Degree?

If you are going to spend the time to study AI/ML then you might as well invest in an online degree which will greatly increase your career opportunities (and a requirement for most all AI engineer positions).

[Do you need a Master’s Degree in Data Science?](https://towardsdatascience.com/do-you-need-a-masters-degree-in-data-science-you-are-asking-the-wrong-question-8c83dec8bf1b?source=rss----7f60cf5620c9---4)


## Recommended Tutorials and Books

See [Tutorials and Books](./how_to_learn.md)


## How to ask an AI/ML question?

1. Describe the dataset in detail and be willing to share your dataset. 

If you are not willing to share the details of the dataset then is impossible for anyone to provide advice. 

Please note that Discord users are more than willing to donate their time to give free consulting advice but it is unethical to try to ask vague questions and get free advice on a commercial or research project that you are getting paid to do. If this is the case then you must mskr s diligent effort to state this fact up front repeatedly (do not expect other Discord users to go data mining for your original post). 

It is usually best to share files via DM  so that other users do not have to search the channels for your files. Keep in mind that Discord channel content is unstructured, so it is difficult to search channels for content. 

2. Avoid asking users to help debug your code. 

In general, the problem is usually not the algorithm implementation but the data preparation and feature engineering of your dataset. If you find yourself mired in debugging code then this should be a red flag that you need to refactor or more likely choose a simpler model (Occam's Razor). 

See [How to Choose an ML Model?](./getting_started.md)

3. Plots and graphs

It is best not to plot more than one metric on a graph since libraries such as matplotlib usually automatically rescale values to better show the difference in values (see [How to Diagnose Overfitting and Underfitting](./ml/diagnose_overfitting.md) for examples).

However, the best practice (and approach used by most ML tools) is to compute the average values of performance metrics (see [Machine Learning Error Metrics](./ml/error_metrics.md) rather than plotting graphs. The only graph that is commonly used in ML is train/val loss to spot-check convergence of training of a model only. 



## How to choose a performance metric?

See [Machine Learning Error Metrics](./ml/error_metrics.md) 


## How to Choose an ML Algorithm?

See [How to Choose an ML Model?](./getting_started.md) and [Understand Machine Learning Algorithms](./ml/error_metrics.md)


## Should I start learning ML by coding an algorithm from scratch?

You need to know what algorithms are available for a given problem, how they work, and how to get the most out of them. However, this does not mean you need to hand-code the algorithms from scratch.

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

