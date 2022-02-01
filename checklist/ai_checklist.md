# AI Checklists

<!-- MarkdownTOC -->

- Feature Engineering
- When You Should not use ML?
- AI Checklist
- Problem Definition Checklist
- Dataset Selection Checklist
- Data Preparation Checklist
- Design Checklist
- Training Checklist
- Operationalize Checklist
- NLP Checklist
- References

<!-- /MarkdownTOC -->

## Feature Engineering

Feature engineering is the process of transforming data to extract valuable information.

If done correctly, feature engineering can play even a bigger role in model performance than hyperparameter tuning.

A checklist for transforming features for better model performance:

[Feature engineering A-Z](https://towardsdatascience.com/feature-engineering-a-z-aa8ce9639632)


The following article explains and implements PCA in Python:

[Dimensionality Reduction Explained](https://towardsdatascience.com/dimensionality-reduction-explained-5ae45ae3058e)

 

## When You Should not use ML?

[4 Reasons Why You Shouldn’t Use Machine Learning](https://towardsdatascience.com/4-reasons-why-you-shouldnt-use-machine-learning-639d1d99fe11?source=rss----7f60cf5620c9---4&gi=204e8d695029)

1. Data-related issues

In the [AI hierarchy of needs](https://hackernoon.com/the-ai-hierarchy-of-needs-18f111fcc007), it is important that you have a robust process for collecting, storing, moving, and transforming data. Otherwise, GIGO. 

Not only do you need your data to be **reliable** but you need **enough** data to leverage the power of machine learning. 

2. Interpretability

There are two main categories of ML models: 

- Predictive models focus on the model’s ability to produce accurate predictions.

- Explanatory models focus more on understanding the relationships between the variables in the data.

ML models (especially ensemble models and neural networks) are predictive models that are much better at predictions than traditional models such as linear/logistic regression.

However, when it comes to understanding the relationships between the predictive variables and the target variable, these models are a black box. You may understand the underlying mechanics behind these models, but it is still not clear how they get to their final results.

In general, ML and deep learning models are great for prediction but lack  explainability.

3. Technical Debt

Maintaining ML models over time can be challenging and expensive. 

There are several types of “debt” to consider when maintaining ML models:

- Dependency debt: The cost of maintaining multiple versions of the same model, legacy features, and underutilized packages.

- Analysis debt: This refers to the idea that ML systems often end up influencing their own behavior if they update over time, resulting in direct and hidden feedback loops.

- Configuration debt: The configuration of ML systems  also incur a debt similar to any software system. 

4. Better Alternatives

ML should not be used when simpler alternatives exist that are equally as effective. 

You should start with the simplest solution that you can implement and iteratively determine if the marginal benefits from the next best alternative outweighs the marginal costs.

Simpler = Better (Occam's Razor)



## AI Checklist

A checklist for transforming features for better model performance

The lifecycle of an AI project can be divided into 6 stages:

1. **Problem definition:** The formative stage of defining the scope, value definition, timelines, governance, resources associated with the deliverable.

2. **Dataset Selection:** This stage can take a few hours or a few months depending on the overall data platform maturity and hygiene. Data is the lifeblood of ML, so getting the right and reliable datasets is supercritical.

3. **Data Preparation:** Real-world data is messy. Understanding data properties and preparing properly can save endless hours down the line in debugging.

4. **Design:** This phase involves feature selection, reasoning algorithms, decomposing the problem, and formulating the right model algorithms.

5. **Training:** Building the model, evaluating with the hold-out examples, and online experimentation.

6. **Operationalize:** This is the post-deployment phase involving observability of the model and ML pipelines, refreshing the model with new data, and tracking success metrics in the context of the original problem.


## Problem Definition Checklist

1. Verify there is quantifiable business value in solving the problem. 

2. Verify that simpler alternatives (such as hand-crafted heuristics) are not sufficient to address the problem.

3. Ensure that the problem has been decomposed into the smallest possible units.

4. Clear understanding on how the AI output will be applied to accomplish the desired business outcome.

5. Clear measurable metric(s) to measure the success of the solution.

6. Clear understanding of precision versus recall tradeoff of the problem.

7. Verify impact when the logistic classification prediction is incorrect.

8. Ensure project costs include the cost of managing corresponding data pipelines.


## Dataset Selection Checklist

1. Verify the meaning of the dataset attributes.

2. Verify the derived metrics used in the project are standardized.

3. Verify data from warehouse is not stale due to data pipeline errors.

4. Verify schema compliance of the dataset.

5. Verify datasets comply with data rights regulations (such as GDPR, CCPA, etc.).

6. Ensure there is a clear change management process for dataset schema changes.

7. Verify dataset is not biased.

8. Verify the datasets being used are not orphaned (without data stewards).


## Data Preparation Checklist

1. Verify data is IID (Independent and Identically Distributed).

2. Verify expired data is not used -- historic data values that may not be relevant.

3. Verify there are no systematic errors in data collection.

4. Verify dataset is monitored for sudden distribution changes.

5. Verify seasonality in data (if applicable) is correctly taken into account.

6. Verify data is randomized (if applicable) before splitting into training and test data.

7. Verify there are no duplicates between test and training examples.

8. Verify sampled data is statistically representative of the dataset as a whole.

9. Verify the correct use of normalization and standardization for scaling feature values.

10. Verify outliers have been properly handled.

11. Verify proper sampling of selected samples from a large dataset.


## Design Checklist

1. Ensure feature crossing is experimented before jumping to non-linear models (if applicable).

2. Verify there is no feature leakage.

3. Verify new features are added to the model with justification documented on how they increase the model quality.

4. Verify features are correctly scaled.

5. Verify simpler ML models are tried before using deep learning.

6. Ensure hashing is applied for sparse features (if applicable).

7. Verify model dimensionality reduction has been explored.

8. Verify classification threshold tuning (in logistic regression) takes into account business impact.

9. Verify regularization or early stopping in logistic regression is applied (if applicable).

10. Apply embeddings to translate large sparse vectors into a lower-dimensional space (while preserving semantic relationships).

11. Verify model freshness requirements based on the problem requirements.

12. Verify the impact of features that were discarded because they only apply to a small fraction of data.

13. Check if feature count is proportional to the amount of data available for model training.


## Training Checklist

1. Ensure interpretability is not compromised prematurely for performance during early stages of model development.

2. Verify model tuning is following a scientific approach (rather than ad-hoc).

3. Verify the learning rate is not too high.

4. Verify root causes are analyzed and documented if the loss-epoch graph is not converging.

5. Analyze specificity versus sparsity trade-off on model accuracy.

6. Verify that reducing loss value improves recall/precision.

7. Define clear criteria for starting online experimentation (canary deployment).

8. Verify per-class accuracy in multi-class classification.

9. Verify infrastructure capacity or cloud budget allocated for training.

10. Ensure model permutations are verified using the same datasets (for an apples-to-apples comparison).

11. Verify model accuracy for individual segments/cohorts, not just the overall dataset.

12. Verify the training results are reproducible -- snapshots of code (algo), data, config, and parameter values.

13. Verify there are no inconsistencies in training-serving skew for features.

14. Verify feedback loops in model prediction have been analyzed.

15. Verify there is a backup plan if the online experiment does not go as expected.

16. Verify the model has been calibrated.

17. Leverage automated hyperparameter tuning (if applicable).

18. Verify prediction bias has been analyzed.

19. Verify dataset has been analyzed for class imbalance.

20. Verify model experimented with regularization lambda to balance simplicity and training data fit.

21. Verify the same test samples are not being used over and over for test and validation.

22. Verify batch size hyperparameter is not too small.

23. Verify initial values in neural networks.

24. Verify the details of failed experiments are captured.

25. Verify the impact of wrong labels before investing in fixing them.

26. Verify a consistent set of metrics are used to analyze the results of online experiments.

27. Verify multiple hyperparameters are not tuned at the same time.


## Operationalize Checklist

1. Verify data pipelines used to generate time-dependent features are performant for low latency.

2. Verify validation tests exist for data pipelines.

3. Verify model performance for the individual data slices.

4. Avoid using two different programming languages between training and deployment.

5. Ensure appropriate model scaling so that inference threshold is within the threshold.

6. Verify data quality inconsistencies are checked at source, ingestion into the lake, and ETL processing.

7. Verify cloud spend associated with the AI product is within budget.

8. Ensure optimization phase to balance quality with model depth and width.

9. Verify monitoring for data and concept drift.

10. Verify unnecessary calibration layers have been removed.

11. Verify there is monitoring to detect slow poisoning of the model due to intermittent errors.


## NLP Checklist

[NLP Cheatsheet](https://medium.com/javarevisited/nlp-cheatsheet-2b19ebcc5d2e)



## References

[AI Checklist](https://towardsdatascience.com/the-ai-checklist-fe2d76907673)

[Machine Learning Performance Improvement Cheat Sheet](https://machinelearningmastery.com/machine-learning-performance-improvement-cheat-sheet/)

[Deploy Your Predictive Model To Production](https://machinelearningmastery.com/deploy-machine-learning-model-to-production/)


