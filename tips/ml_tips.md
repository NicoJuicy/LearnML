# Machine Learning Tips

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



## Hyperparameter Tuning 

The process of searching for optimal hyperparameters is called hyperparameter tuning or hypertuning, and is essential in any machine learning project. Hypertuning helps boost performance and reduces model complexity by removing unnecessary parameters (e.g., number of units in a dense layer).

There are two types of hyperparameters:

- Model hyperparameters: influence model architecture (such as number and width of hidden layers in a DNN)

- Algorithm hyperparameters: influence the speed and quality of training (such as learning rate and activation function).


A practical guide to hyperparameter optimization using three methods: grid, random and bayesian search (with skopt)

1. Introduction to hyperparameter tuning.
2. Explanation about hyperparameter search methods.
3. Code examples for each method.
4. Comparison and conclusions.

### Simple pipeline used in all the examples

```py
    from sklearn.pipeline import Pipeline #sklearn==0.23.2
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import make_column_transformer
    from lightgbm import LGBMClassifier
    
    tuples = list()
    
    tuples.append((Pipeline([
            ('scaler', StandardScaler()),
        ]), numeric_var))
    
    tuples.append((Pipeline([
            ('onehot', OneHotEncoder()),
        ]), categorical_var))
    
    preprocess = make_column_transformer(*tuples)
    
    pipe = Pipeline([
        ('preprocess', preprocess),
        ('classifier', LGBMClassifier())
        ])
```


### Grid Search

The basic method to perform hyperparameter tuning is to try all the possible combinations of parameters.

### Random Search

in randomized search, only part of the parameter values are evaluated. 

The parameter values are sampled from a given list or specified distribution. 

The number of parameter settings that are sampled is given by `n_iter`. 

Sampling without replacement is performed when the parameters are presented as a list (similar to grid search), but if the parameter is given as a distribution then sampling with replacement is used (recommended).

The advantage of randomized search is that you can extend your search limits without increasing the number of iterations. You can also use random search to find narrow limits to continue a thorough search in a smaller area.

### Bayesian Search

The main difference with Bayesian search is that the algorithm optimizes its parameter selection in each round according to the previous round score. Thus, the algorithm optimizes the choice and theoretically reaches the best parameter set faster than the other methods which means that this method will choose only the relevant search space and discard the range of values that will most likely not deliver the best solution. 

Thus, Bayesian search can be beneficial when you have a large amount of data and/or the learning processis slow and you want to minimize the tuning time.

```py
    from skopt import BayesSearchCV
    
    # Bayesian
    n_iter = 70
    
    param_grid = {
        "classifier__learning_rate": (0.0001, 0.1, "log-uniform"),
        "classifier__n_estimators": (100,  1000) ,
        "classifier__max_depth": (4, 400) 
    }
    
    reg_bay = BayesSearchCV(estimator=pipe,
                        search_spaces=param_grid,
                        n_iter=n_iter,
                        cv=5,
                        n_jobs=8,
                        scoring='roc_auc',
                        random_state=123)
    
    model_bay = reg_bay.fit(X, y)
```


### Visualization of parameter search (learning rate)

### Visualization of the mean score for each iteration



## References

[A Practical Introduction to Grid Search, Random Search, and Bayes Search](A Practical Introduction to Grid Search, Random Search, and Bayes Search)

[Hyperparameter Tuning Methods](https://towardsdatascience.com/bayesian-optimization-for-hyperparameter-tuning-how-and-why-655b0ee0b399)

[Hyperparameter Tuning with KerasTuner and TensorFlow](https://towardsdatascience.com/hyperparameter-tuning-with-kerastuner-and-tensorflow-c4a4d690b31a)


