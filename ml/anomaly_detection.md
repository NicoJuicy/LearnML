# Anomaly Detection

## Summary

- Beyond accuracy, the **False Positive** and **False Negative rates** are ways of assessing performance

- **Not all anomaly detectors are equal:** performance scores can differ substantially between anomaly detectors, operating on the same real-life time-series data for business metrics

- In our test data, Avora’s anomaly detector achieves better performance compared to Facebook Kats, with significantly lower False Positive & Negative rates but comparable accuracy

- **Even lower False Positive/Negative Rates** can be achieved with hyper-parameters tuning with no reduction in accuracy

In Avora we created an evaluation pipeline using a real life, time-series based on business data to benchmark **Avora** performance against the well known Facebook **Kats** Anomaly Detector which is closely related to the popular Facebook Prophet package.

## Intuitively Measuring and Explaining Performance

Beyond accuracy, the most commonly used metrics when evaluating anomaly detection solutions are F1, Precision, and Recall. 

We can think about these metrics in the following way:

- **Recall** is used to answer the question: What proportion of true anomalies was identified?

- **Precision** answers the question: What proportion of identified anomalies are true anomalies?

- **F1 Score** identifies the overall performance of the anomaly detection model by combining both Recall and Precision using the harmonic mean. 

## False Positive and False Negative Rates

When deciding on the anomaly detection system it is important to pay attention to metrics: False Positive and False Negative rates.

**False Positive** rate helps you understand how many times, on average, will your detector cry wolf and flag the data points that are actually not true anomalies.

Pick the system with the lowest possible False Positive rate. 

If the False Positive rate is too high, the users will turn off the system since it is more distracting than useful.

**False Negative** rate shows how many anomalies were, on average, missed by the detector.

Choose the system with the lowest possible False Negatives rate. 

If the False Negative rate is too high, you will be missing a lot of crucial anomalies and in time you will lose trust in the system.

## Methodology

For the performance comparison we created a system that aimed to provide an objective, unbiased evaluation.

  1. Time-series data from real-life examples was collected & anonymised.

  2. Anomalies were manually labelled prior to performance evaluation which was stored as the **ground truth dataset** based on the analyst’s assessment, independent of results from either algorithm.

  3. The ground truth dataset was used by a pipeline which only performed the evaluation after both Avora and KATS anomaly detection algorithms completed the labeling.


## References

[Anomaly Detection — How to Tell Good Performance from Bad](https://towardsdatascience.com/anomaly-detection-how-to-tell-good-performance-from-bad-b57116d71a10)



