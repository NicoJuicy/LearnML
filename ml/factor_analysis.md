## FA vs PCA vs DRO

**Factor analysis (FA** is one of the most well-known methods of classical multivariate analysis []. FA is useful when a large number of variables are believed to be determined by a relatively few common causes or factors. In short, FA involves the calculation of a variable-by-variable correlation matrix that is used to extracts new variables which are a linear combination of the original variables; the coefficients in each linear combination are known as factor loadings which can be used to identify the variables that are most closely related to a factor []. Thus, the factors extracted are the maximum likelihood estimates of the factor loadings. Perhaps the most widely used method for determining the number of factors is using eigenvalues greater than one [59].

However, FA is often confused with principal components analysis (PCA) []. The two methods are related since factor analysis is essentially equivalent to principal components analysis if the errors in the factor analysis model are assumed to all have the same variance. For example, Alama and Baulkani [] applied factor analysis for the NYSE, DJIA and S&P500 indices and were able to reduce eight technical indicators (columns or variables) to three factors. 

Lv, Wang, Li, and Xiang [87] state that dimensionality reduction operation (DRO) is primarily used to deal with the high-dimensionality of stock data but there have been no studies comparing the performance of DNN models based on different DRO techniques. Therefore, the authors applied four of the most commonly used DRO:  principal component analysis (PCA), least absolute shrinkage and selection operator (LASSO), classification and regression trees (CART), and autoencoder (AE) to analyze the prediction performance of six popular DNN models: MLP, Deep Belief Network (DBN), Stacked Auto-Encoders (SAE), RNN, LSTM, and Gated Recurrent Unit (GRU). The study used the following performance measure indicators: winning ratio (WR), annualized return rate (ARR), annualized Sharpe ratio (ASR), and maximum drawdown (MDD) and the execution speed indicator which is average training time of generating the trading signals of an individual stock (ATT). 

The study evaluated 424 S&P 500 index component stocks (SPICS) and 185 CSI 300 index component stocks (CSICS) and concluded the following:

1. There was no significant difference in performance of the algorithms 
2. DRO does not significantly improve the execution speed of any of the DNN models.


