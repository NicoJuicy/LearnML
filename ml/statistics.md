# Statistics

[Important Statistics Data Scientists Need to Know](https://www.kdnuggets.com/2021/09/important-statistics-data-scientists.html)

Here is a list of the topics covered in the article:

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


## Correlation

[Statistics in Python — Understanding Variance, Covariance, and Correlation](https://towardsdatascience.com/statistics-in-python-understanding-variance-covariance-and-correlation-4729b528db01?source=rss----7f60cf5620c9---4)


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


## References

[What does RMSE really mean?](https://towardsdatascience.com/what-does-rmse-really-mean-806b65f2e48e)

[Reasoning under Uncertainty (Chapters 13 and 14.1 - 14.4)](http://pages.cs.wisc.edu/~dyer/cs540/notes/uncertainty.html)

[Conditional independence in general](http://www.cs.columbia.edu/~kathy/cs4701/documents/conditional-independence-bn.txt)
