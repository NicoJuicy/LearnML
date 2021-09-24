# Long Short-Term Memory Networks (LSTMs)

[Guide to Long Short-Term Memory Networks (LSTMs)](https://machinelearningmastery.com/start-here/#deeplearning)

Long Short-Term Memory (LSTM) Recurrent Neural Networks are designed for sequence prediction problems and are a state-of-the-art deep learning technique for challenging prediction problems.

[Making Predictions with Sequences](https://machinelearningmastery.com/sequence-prediction/)

[A Gentle Introduction to Long Short-Term Memory Networks](https://machinelearningmastery.com/gentle-introduction-long-short-term-memory-networks-experts/)

[Introduction to Models for Sequence Prediction](https://machinelearningmastery.com/models-sequence-prediction-recurrent-neural-networks/)


[The 5 Step Life-Cycle for Long Short-Term Memory Models in Keras](https://machinelearningmastery.com/5-step-life-cycle-long-short-term-memory-models-keras/)


## Data Preparation for LSTM

[How to Reshape Input Data for Long Short-Term Memory Networks](https://machinelearningmastery.com/reshape-input-data-long-short-term-memory-networks-keras/)

[How to One Hot Encode Sequence Data](https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/)

[How to Remove Trends and Seasonality with a Difference Transform](https://machinelearningmastery.com/remove-trends-seasonality-difference-transform-python/)

[How to Scale Data for Long Short-Term Memory Networks](https://machinelearningmastery.com/how-to-scale-data-for-long-short-term-memory-networks-in-python/)

How to Prepare Sequence Prediction for Truncated BPTT

How to Handle Missing Timesteps in Sequence Prediction Problems


## LSTM Behaviour

A Gentle Introduction to Backpropagation Through Time

Demonstration of Memory with a Long Short-Term Memory Network

How to Use the TimeDistributed Layer for Long Short-Term Memory Networks

How to use an Encoder-Decoder LSTM to Echo Sequences of Random Integers

Attention in Long Short-Term Memory Recurrent Neural Networks


## Modeling with LSTM

Generative Long Short-Term Memory Networks
Stacked Long Short-Term Memory Networks

Encoder-Decoder Long Short-Term Memory Networks

CNN Long Short-Term Memory Networks

[Diagnose Overfitting and Underfitting of LSTM Models](https://machinelearningmastery.com/diagnose-overfitting-underfitting-lstm-models/)

[How to Make Predictions with Long Short-Term Memory Models](https://machinelearningmastery.com/make-predictions-long-short-term-memory-models-keras/)


## LSTM for Time Series

[On the Suitability of LSTMs for Time Series Forecasting](https://machinelearningmastery.com/suitability-long-short-term-memory-networks-time-series-forecasting/)

[Time Series Forecasting with the Long Short-Term Memory Network](https://machinelearningmastery.com/time-series-forecasting-long-short-term-memory-network-python/)

[Multi-step Time Series Forecasting with Long Short-Term Memory Networks](https://machinelearningmastery.com/multi-step-time-series-forecasting-long-short-term-memory-networks-python/)

[Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)


## LSTM Autoencoders

[A Gentle Introduction to LSTM Autoencoders](https://machinelearningmastery.com/lstm-autoencoders/)

An **LSTM Autoencoder** is an implementation of an autoencoder for sequence data using an Encoder-Decoder LSTM architecture.

The encoder part of the model can be used to encode or compress sequence data that in turn may be used in data visualizations or as a feature vector input to a supervised learning model.

- Autoencoders are a type of self-supervised learning model that can learn a compressed representation of input data.

- LSTM Autoencoders can learn a compressed representation of sequence data and have been used on video, text, audio, and time series sequence data.

This post is divided into six sections; they are:

1. What Are Autoencoders?
2. A Problem with Sequences
3. Encoder-Decoder LSTM Models
4. What Is an LSTM Autoencoder?
5. Early Application of LSTM Autoencoder
6. How to Create LSTM Autoencoders in Keras


[How to Develop an Encoder-Decoder Model for Sequence-to-Sequence Prediction in Keras](https://machinelearningmastery.com/develop-encoder-decoder-model-sequence-sequence-prediction-keras/)

The encoder-decoder model provides a pattern for using recurrent neural networks to address challenging sequence-to-sequence prediction problems such as machine translation.

Encoder-decoder models can be developed in the Keras Python deep learning library and an example of a neural machine translation system developed with this model has been described on the Keras blog, with sample code distributed with the Keras project.


## Encoder-Decoder with Attention

[Implementation Patterns for the Encoder-Decoder RNN Architecture with Attention](https://machinelearningmastery.com/implementation-patterns-encoder-decoder-rnn-architecture-attention/)

The encoder-decoder model for recurrent neural networks is an architecture for sequence-to-sequence prediction problems where the length of input sequences is different to the length of output sequences.

It is comprised of two sub-models:

- Encoder: The encoder is responsible for stepping through the input time steps and encoding the entire sequence into a fixed length vector called a context vector.

- Decoder: The decoder is responsible for stepping through the output time steps while reading from the context vector.

A problem with the architecture is that performance is poor on long input or output sequences. The reason is believed to be because of the fixed-sized internal representation used by the encoder.

_Attention_ is an extension to the architecture that addresses this limitation by providing a richer context from the encoder to the decoder and a learning mechanism where the decoder can learn where to pay attention in the richer encoding when predicting each time step in the output sequence.


—————————-


## Deep Learning for Time Series Forecasting

Deep learning neural networks are able to automatically learn arbitrary complex mappings from inputs to outputs and support multiple inputs and outputs.

In fact, models such as MLPs, CNNs, and LSTMs offer a lot of promise for time series forecasting.

[Guide to Deep Learning for Time Series Forecasting](https://machinelearningmastery.com/start-here/#deep_learning_time_series)

[Taxonomy of Time Series Forecasting Problems](https://machinelearningmastery.com/taxonomy-of-time-series-forecasting-problems/)

[How to Develop a Skillful Machine Learning Time Series Forecasting Model](https://machinelearningmastery.com/how-to-develop-a-skilful-time-series-forecasting-model/)

[Comparing Classical and Machine Learning Algorithms for Time Series Forecasting](https://machinelearningmastery.com/findings-comparing-classical-and-machine-learning-methods-for-time-series-forecasting/)


### Model Types

How to Develop MLPs for Time Series Forecasting

[How to Develop CNNs for Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-convolutional-neural-network-models-for-time-series-forecasting/)

[How to Develop LSTMs for Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/)


### Human Activity Recognition (multivariate classification)

How to Model Human Activity From Smartphone Data

How to Develop CNN Models for Human Activity Recognition

How to Develop RNN Models for Human Activity Recognition


### Forecast Electricity Usage (multivariate, multi-step)

How to Load and Explore Household Electricity Usage Data

[Multi-step Time Series Forecasting with Machine Learning](https://machinelearningmastery.com/multi-step-time-series-forecasting-with-machine-learning-models-for-household-electricity-consumption/)

[How to Develop CNNs for Multi-Step Time Series Forecasting](https://machinelearningmastery.com/how-to-develop-convolutional-neural-networks-for-multi-step-time-series-forecasting/)


### Forecast Air Pollution (multivariate, multi-step)

[Load, Visualize, and Explore a Air Pollution Forecasting](https://machinelearningmastery.com/how-to-load-visualize-and-explore-a-complex-multivariate-multistep-time-series-forecasting-dataset/)

[Develop Baseline Forecasts for Air Pollution Forecasting](https://machinelearningmastery.com/how-to-develop-baseline-forecasts-for-multi-site-multivariate-air-pollution-time-series-forecasting/)

[Develop Autoregressive Models for Air Pollution Forecasting](https://machinelearningmastery.com/how-to-develop-autoregressive-forecasting-models-for-multi-step-air-pollution-time-series-forecasting/)

[Develop Machine Learning Models for Air Pollution Forecasting](https://machinelearningmastery.com/how#to-develop-machine-learning-models-for-multivariate-multi-step-air-pollution-time-series-forecasting/)



## Tuning

[How to Grid Search Hyperparameters for Deep Learning Models in Python With Keras](https://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/)

[A Practical Introduction to Grid Search, Random Search, and Bayes Search](https://towardsdatascience.com/a-practical-introduction-to-grid-search-random-search-and-bayes-search-d5580b1d941d?gi=c4f3f0ee4378)

[Hyperparameter Tuning Methods - Grid, Random or Bayesian Search?](https://towardsdatascience.com/bayesian-optimization-for-hyperparameter-tuning-how-and-why-655b0ee0b399?gi=68cca7c0df4b)

