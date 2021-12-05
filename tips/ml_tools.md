# Machine Learning Tools

Here is a list of ML tools that I have found to be helpful to an AI/ML engineer. 

<!-- MarkdownTOC levels=1,2 -->

- How to choose an ML framework?
- Keras Tutorials
- Data Exploration Tools
- Feature Engineering Tools
- ML Libraries
- Kale
- Kedro
- Computer Vision
- Natural Language Programming
- Python Libraries
- Cloud Tools
- Deep Learning Tools
- Linux Utilities
- Time Series Libraries
- References

<!-- /MarkdownTOC -->


## How to choose an ML framework?

[Keras vs PyTorch for Deep Learning](https://towardsdatascience.com/keras-vs-pytorch-for-deep-learning-a013cb63870d)

## Keras Tutorials

[Introduction to Keras for Engineers](https://keras.io/getting_started/intro_to_keras_for_engineers/)

[3 ways to create a Machine Learning model with Keras and TensorFlow 2.0 (Sequential, Functional, and Model Subclassing)](https://towardsdatascience.com/3-ways-to-create-a-machine-learning-model-with-keras-and-tensorflow-2-0-de09323af4d3)

[Introducing TensorFlow Datasets](https://medium.com/tensorflow/introducing-tensorflow-datasets-c7f01f7e19f3)

[Getting started with TensorFlow 2.0](https://medium.com/@himanshurawlani/getting-started-with-tensorflow-2-0-faf5428febae)

[Introducing TensorFlow Datasets](https://medium.com/tensorflow/introducing-tensorflow-datasets-c7f01f7e19f3)

[How to (quickly) Build a Tensorflow Training Pipeline](https://towardsdatascience.com/how-to-quickly-build-a-tensorflow-training-pipeline-15e9ae4d78a0?gi=f2df1cc3455f)


## Data Exploration Tools

### [Orange](https://github.com/biolab/orange3)

[Orange Docs](https://orangedatamining.com/docs/)

Orange is a data mining and visualization toolbox for novice and expert alike

A great tool for exploratory data analysis (EDA) and to evaluate many models as you narrow the choices for a particular dataset.

### [dataprep](https://github.com/sfu-db/dataprep)

DataPrep lets you prepare your data using a single library with a few lines of code.

DataPrep.EDA is the fastest and the easiest EDA (Exploratory Data Analysis) tool in Python which allows you to understand a Pandas/Dask DataFrame with a few lines of code in seconds.

**Create Profile Reports, Fast**

You can create a beautiful profile report from a Pandas/Dask DataFrame with the create_report function. DataPrep.EDA has the following advantages compared to other tools:

10X Faster: DataPrep.EDA can be 10X faster than Pandas-based profiling tools due to its highly optimized Dask-based computing module.
Interactive Visualization: DataPrep.EDA generates interactive visualizations in a report, which makes the report look more appealing to end users.
Big Data Support: DataPrep.EDA naturally supports big data stored in a Dask cluster by accepting a Dask dataframe as input.

### [Bamboolib](https://github.com/8080labs/bamboolib)

Bamboolib is a data analysis and exploration library that is very easy to use plus you can see the code and reuse it. 

### [TensorFlow Data Validation](https://github.com/tensorflow/data-validation)

TensorFlow Data Validation (TFDV) makes data exploration for machine learning more intuitive

TFDV allows you to get a quick look at the statistics of a dataset.

TFDV can also identify anomalies in the data and can be configured to detect different classes of anomalies.

### [Great Expectations](https://github.com/great-expectations/great_expectations) 

Great Expectations is a Python library that helps us validate, document, and profile our data so that we always make sure it is good and just like we expect it to be.

Great Expectations provides several functions to evaluate the data from many different perspectives.

[A Great Python Library: Great Expectations](https://towardsdatascience.com/a-great-python-library-great-expectations-6ac6d6fe822e)

The article uses the Great Expectations library to evaluate a sample dataset.



## Feature Engineering Tools

There are many tools that will help you in automating the entire feature engineering process and producing a large pool of features in a short period of time for both classification and regression tasks.

### [Feature-engine](https://github.com/feature-engine/feature_engine)

Feature-engine is a Python library with multiple transformers to engineer and select features for use in machine learning models. 

Feature-engine's transformers follow scikit-learn's functionality with fit() and transform() methods to first learn the transforming parameters from data and then transform the data.

### [Featuretools](https://github.com/alteryx/featuretools)

Featuretools is a framework to perform automated feature engineering that excels at transforming temporal and relational datasets into feature matrices for machine learning. 

Featuretools integrates with the machine learning pipeline-building tools you already have. 

In a fraction of the time it would take to do it manually, you can load in pandas dataframes and automatically construct significant features.

### AutoFeat

AutoFeat helps to perform Linear Prediction Models with Automated Feature Engineering and Selection. 

AutoFeat allows you to select the units of the input variables in order to avoid the construction of physically nonsensical features.

[The Only Web Scraping Tool you need for Data Science](https://medium.com/nerd-for-tech/the-only-web-scraping-tool-you-need-for-data-science-f388e2afa187)



## ML Libraries

### [OpenCV](https://docs.opencv.org/master/d9/df8/tutorial_root.html)

[Introduction to OpenCV](https://www.geeksforgeeks.org/introduction-to-opencv/)

[OpenCV Python Tutorial](https://www.geeksforgeeks.org/opencv-python-tutorial/)

OpenCV is a huge open-source library for computer vision, machine learning, and image processing. 

OpenCV supports a wide variety of programming languages like Python, C++, Java, etc. 

OpenCV can process images and videos to identify objects, faces, or eve handwriting.


## [Kale](https://github.com/kubeflow-kale/kale)

[How to start contributing to open-source projects](https://towardsdatascience.com/how-to-start-contributing-to-open-source-projects-41fcfb654b2e)

KALE (Kubeflow Automated pipeLines Engine) is an open-source project that simplifies the Data Science experience with Kubeflow and Kubernetes (deploying Kubeflow Pipelines workflows). 


## [Kedro](https://github.com/quantumblacklabs/kedro)

### [PyMC3](https://github.com/pymc-devs/pymc3)

PyMC3 is a Python package for Bayesian statistical modeling and Probabilistic Machine Learning focusing on advanced Markov chain Monte Carlo (MCMC) and variational inference (VI) algorithms. 

The flexibility and extensibility of PyMC3 make it applicable to a large suite of problems.

[A Gentle Introduction to Bayesian Belief Networks](https://machinelearningmastery.com/introduction-to-bayesian-belief-networks/)

[Building DAGs with Python](https://mungingdata.com/python/dag-directed-acyclic-graph-networkx/)

[bnlearn](https://github.com/erdogant/bnlearn)


### [MXNet](https://github.com/apache/incubator-mxnet)

MXNet is an open-source framework designed to train and deploy deep neural networks with a strong focus on scalability. 

In contrast to the torch and tflow￼ frameworks, MXNet is a child of the Apache foundation and probably the best true open-source framework. 

A rising star and already accessible in many languages, not just Python.

It is not as popular but still supported by all major clouds and can potentially speed up your computations.

### [ONNX](https://github.com/onnx/onnx)

ONNX is a format to represent machine learning models. The goal is that future developers are no longer forced to pick a model that has been trained in their particular framework.

ONNX achieves this goal by using a common set of operators (the mathematical operations) that happen between the different layers and a common file format. 

ONNX is backed by pretty most of the big AI players and is sure to have a huge impact on the future of ML development. Apart from Google all of the tech giants are partners in this project.


### [openai/gym](https://github.com/openai/gym)

An open-source toolkit for developing and comparing reinforcement learning algorithms. It is compatible with any numerical computation lib



## Computer Vision

### [ageitgey/face_recognition](https://github.com/ageitgey/face_recognition)

A library for building face recognition applications that is one of the simplest facial recognition APIs for Python and the command line.



## Natural Language Programming

### [NLTK](https://www.nltk.org)

[NLTK Book](https://www.nltk.org/book/)

Natural language toolkit (NLTK) is by far the most popular platform for building NLP related projects.

NLTK is also an open-source library and is available for almost every kind of operating system. 

### [Polyglot](https://github.com/aboSamoor/polyglot)

Polyglot is a python library for NLP that supports extensive multilingual applications. 

According to the documentation, polyglot supports tokenization for 165 languages, language detection for 196 languages, part-of-speech tagging for 16 languages and sentiment analysis in more than 130 languages.

### [SpaCy](https://spacy.io/)

SpaCy is a Python NLP library that can be useful for real-world projects containing huge amounts of text data.

The main advantage of SpaCy is that it is much faster than other libraries since it is written in Cython which makes it capable of handling larger amounts of data efficiently.

spaCy comes with pretrained pipelines and currently supports tokenization and training for 60+ languages with pre-trained transformers such as BERT and support for modern ML/DL frameworks such as Pytorch and Tensorflow which makes SpaCy a good choice for professional projects.

### [GenSim](https://radimrehurek.com/gensim/)

GenSim is an NLP library that is popular due to its amazing speed and memory optimization. 

All the libraries used in GenSim are memory independent and can run easily with data sets of large size also. 

GenSim uses SciPy and NumPy for computing and is used in applications such as chatbots, semantic search, etc.

### [Pattern](https://github.com/clips/pattern)

Pattern is a multipurpose Python library that can be used for different tasks such as natural language processing (tokenization, sentiment analysis, POS tagging, etc.) as well as data mining from websites and machine learning using built-in models such as K-nearest Neighbors, Support Vector Machine, etc.

This library is easy to understand and implement for beginners due to its simple and straightforward syntax and it is also helpful for web developers who need to work with text data.

### [clean-text](https://github.com/jfilter/clean-text)

User-generated content on the Web and in social media is often dirty. Preprocess your scraped data with clean-text to create a normalized text representation.



## Python Libraries

### [dateutil](https://github.com/dateutil/dateutil)

The dateutil module provides powerful extensions to the standard datetime module.

### gRPC

The trend now is toward gRPC for microservices since it is more secure, faster, and more robust (especially with IoT).

[gRPC with REST and Open APIs](https://grpc.io/blog/coreos/)

- gRPC was recommended for developing microservices by my professor in Distributed Computing course.

- gRPC uses HTTP/2 which enables applications to present both a HTTP 1.1 REST/JSON API and an efficient gRPC interface on a single TCP port (available for Go). 

- gRPC provides developers with compatibility with the REST web ecosystem while advancing a new, high-efficiency RPC protocol. 

- With the recent release of Go 1.6, Go ships with a stable net/http2 package by default.


### Pickle

Pickle can be used to save and load the python classes or PyTorch models for reuse.


### Plots and Graphs

Matplolib is the basis of image visualization in Python but there are other alternatives such as seaborn and plotly (which is browser-based).


### [PySpark](https://github.com/apache/spark)

PySpark is similar to pandas but built with distributed computing in mind. 

PySpark is also focused on massive parallel Machine Learning for your very big data use cases. 

### [tqdm](https://github.com/tqdm/tqdm)

tqdm gives you a progress bar that you can use on any `for` loop that tells you how long each iteration takes on average and how long it will take. 

### [urllib3](https://github.com/urllib3/urllib3)

urllib3 is a Python HTTP library with thread-safe connection pooling, file post support, user friendly, and more.

In short, urllib3 is a powerful, user-friendly HTTP client for Python.


### [The Algorithms - Python](https://github.com/TheAlgorithms/Python)

All algorithms implemented are for learning purposes. 

### [Awesome Python Awesome](https://github.com/vinta/awesome-python)

A curated list of awesome Python frameworks, libraries, software, and resources.



## Cloud Tools

### [Deta](https://www.deta.sh)

Deta is a free, developer friendly cloud platform. 

**Deta  Micros** is a service to deploy Python and Node.js apps/APIs on the internet in seconds. 

**Deta Base** is a super easy to use production-grade NoSQL database that comes with unlimited storage.

**Deta Drive** is an easy to use cloud storage solution by Deta – get 10GB for free



## Deep Learning Tools

[Kedro — A Python Framework for Reproducible Data Science Project](https://towardsdatascience.com/kedro-a-python-framework-for-reproducible-data-science-project-4d44977d4f04)

Kedro is an open-source Python framework for creating reproducible, maintainable, and modular data science code. 

Kedro borrows concepts from software engineering and applies them to machine-learning code such as modularity, separation of concerns, and versioning.

### [Poetry](https://python-poetry.org/)

[Are You Still Using Virtualenv for Managing Dependencies in Python Projects?](https://towardsdatascience.com/poetry-to-complement-virtualenv-44088cc78fd1)

Poetry is a tool for dependency management and packaging in Python.

### [Hydra](https://hydra.cc/docs/intro/)

Hydra provides a configuration file for the entire experiment. 

Hydra is helpful when we want to share your code with someone else or run the experiments on a different machine. 

Hydra provides the flexibility to set the desired configurations like learning rate, model hidden layer sizes, epochs, data set name, etc. without having to make changes to the actual code.

### [Pipreqs](https://pypi.org/project/pipreqs/)

Pipreqs is useful when you want to port your code to a different machine and install all the dependencies. 

Pipreqs helps to create a list of python dependencies along with the versions that the current working code is using and saves them in a file that can be easily installed anywhere.

### [Loguru](https://loguru.readthedocs.io/en/stable/api/logger.html)

Loguru provides the functionality of a logger to log configuration, experiment name, and other training-related data.

Loguru provides the functionality of a logger to log configuration, experiment name, and other training-related data. 

### [H5py](https://docs.h5py.org/en/stable/quick.html)

H5py can be used to store all the intermediate loss values in a dictionary mapped to appropriate key which can be loaded to be reused as a python code.

### [Tqdm](https://tqdm.github.io/)

Tqdm can be used with a loop (such as a loop over a torch `DataLoader` object) to provide a picture of time per gradient step or epoch which can help in setting logging frequency of different results, saving the model, or getting an idea of how to set the validation intervals.



## Linux Utilities

### Lucidchart

Lucidchart is a diagraming tool that also has shared space for collaboration and the ability to make notes next to diagrams. 

### Streamlit

Streamlit turns data scripts into shareable web apps in minutes. 

Streamlit is one of the fastest way to build and share data apps. 

All in Python. All for free. No front‑end experience required.

### [Screen](https://linuxize.com/post/how-to-use-linux-screen/)

Screen is a GNU linux utility that lets you launch and use multiple shell sessions from a single ssh session. The process started with screen can be detached from session and then reattached at a later time. So your experiments can be run in the background, without the need to worry about session closing, or terminal crashing.

### tmux

[How To Install And Use tmux On Ubuntu 12.10](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-tmux-on-ubuntu-12-10--2)

tmux is a terminal multiplexer that allows you to access a tmux terminal using multiple virtual terminals.

This means that you can run several terminals at once concurrently off of a single tmux session without spawning any new actual terminal sessions.

This also means that sudden disconnects from a cloud server running tmux will not kill the processes running inside the tmux session.



## Time Series Libraries

### [statsmodels](https://github.com/statsmodels/statsmodels/)

statsmodels is a Python package that provides a complement to scipy for statistical computations including descriptive statistics and estimation and inference for statistical models.

### [stumpy](https://github.com/TDAmeritrade/stumpy)

STUMPY is a powerful and scalable library that efficiently computes something called the matrix profile which can be used for a variety of time series data mining tasks. 

### [Darts](https://github.com/unit8co/darts)

Darts is a Python library for easy manipulation and forecasting of time series that contains a variety of models, from classics such as ARIMA to deep neural networks. 

The models can all be used in the same way using fit() and predict() functions which is similar to scikit-learn. 

The library claims to make it easy to backtest models and combine the predictions of several models and external regressors. 

Darts supports both univariate and multivariate time series and models. 

The neural networks can be trained on multiple time series, and some of the models offer probabilistic forecasts.

### [AutoTS](https://github.com/winedarksea/AutoTS)

Forecasting Model Selection for Multiple Time Series

AutoML for forecasting with open-source time series implementations.

### [TsFresh](https://github.com/blue-yonder/tsfresh)

tsfresh is a python package that calculates a huge number of time series characteristics or features automatically. 

In addition, the package includes methods for assessing the explanatory power and significance of such traits in regression and classification tasks.



## References

[All Top Python Libraries for Data Science Explained](https://towardsdatascience.com/all-top-python-libraries-for-data-science-explained-with-code-40f64b363663)

[7 Amazing Python Libraries For Natural Language Processing](https://towardsdatascience.com/7-amazing-python-libraries-for-natural-language-processing-50ca6f9f5f11)

[4 Amazing Python Libraries That You Should Try Right Now](https://towardsdatascience.com/4-amazing-python-libraries-that-you-should-try-right-now-872df6f1c93)

[Colaboratory FAQ](https://research.google.com/colaboratory/faq.html#resource-limits)

[Tools for Efficient Deep Learning](https://towardsdatascience.com/tools-for-efficient-deep-learning-c9585122ded0)
