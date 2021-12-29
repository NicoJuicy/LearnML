# AutoML Tools

Automated Machine Learning (AutoML) is an emerging field in which the process of building machine learning models to model data is automated.

Here is a list of AutoML tools that I have found to be helpful to an AI/ML engineer. 

<!-- MarkdownTOC -->

- Quick Start
- MLOps Tools
    - wandb
    - MLFlow
- AutoML Tools
    - PyCaret
    - H2O
    - auto-sklearn
    - AutoGluon
    - AutoKeras
    - Auto-ViML
    - AutoViz
    - MediaPipe
    - Ray
    - FLAML
    - SageMaker
- Time Series Libraries

<!-- /MarkdownTOC -->

## Quick Start

- PyCaret
- auto-sklearn
- Auto-ViML
- AutoViz
- H2O


- MLFlow
- wandb


## MLOps Tools

### [wandb](https://docs.wandb.ai/)

Weights and biases (W&B/wandb) lets you track, compare, visualize, and optimize machine learning experiments with just a few lines of code. 

wandb also lets you track your datasets. 

Using W&B's lightweight tools, you can quickly track experiments, version and iterate on datasets, evaluate model performance, reproduce models, visualize results and spot regressions, and share findings with colleagues. 


### [MLFlow](https://github.com/mlflow/mlflow)

[Tutorial](https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)

Similar to W&B, MLFlow provides functionality for logging code, models, and datasets on which your model has been trained. 

MLflow is a platform to streamline machine learning development including tracking experiments, packaging code into reproducible runs, and deploying models. 

MLflow offers a set of lightweight APIs that can be used with any existing machine learning application or library (TensorFlow, PyTorch, XGBoost, etc.) wherever you currently run ML code (such as notebooks, standalone applications, or the cloud). 




## AutoML Tools

### [PyCaret](https://github.com/pycaret/pycaret)

PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. 

PyCaret is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive.

In comparison with the other open-source machine learning libraries, PyCaret is a low-code library that can be used to replace hundreds of lines of code with only a few words. 

PyCaret is essentially a Python wrapper for several machine learning libraries and frameworks such as scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray, and more.


### H2O

[h2oai/h2o-3](https://github.com/h2oai/h2o-3)

H2O is an Open Source, Distributed, Fast and Scalable ML Platform. 

[H2O Docs](https://docs.h2o.ai/)

[H2O Tutorials](https://github.com/h2oai/h2o-tutorials)

[H2O Deep Learning](https://github.com/h2oai/h2o-tutorials/tree/master/tutorials/deeplearning)

[Automated Machine Learning with H2O (towardsdatascience)](https://towardsdatascience.com/automated-machine-learning-with-h2o-258a2f3a203f)

H2O is an in-memory platform for distributed, scalable machine learning. 

H2O uses familiar interfaces auch as R, Python, Scala, Java, JSON and the Flow notebook/web interface and works seamlessly with big data technologies like Hadoop and Spark. 

H2O provides implementations of many popular algorithms such as Generalized Linear Models (GLM), Gradient Boosting Machines (including XGBoost), Random Forests, Deep Neural Networks, Stacked Ensembles, Naive Bayes, Generalized Additive Models (GAM), Cox Proportional Hazards, K-Means, PCA, Word2Vec, as well as a fully automatic machine learning algorithm (H2O AutoML).

H2O is extensible so that developers can add data transformations and custom algorithms of their choice and access them through all of those clients. 

H2O models can be downloaded and loaded into H2O memory for scoring or exported into POJO or MOJO format for extemely fast scoring in production.



### [auto-sklearn](https://github.com/automl/auto-sklearn)

auto-sklearn is an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator.

[Auto-Sklearn for Automated Machine Learning in Python](https://machinelearningmastery.com/auto-sklearn-for-automated-machine-learning-in-python/)


### [AutoGluon](https://github.com/awslabs/autogluon)

AutoGluon automates machine learning tasks enabling you to easily achieve strong predictive performance in your applications. 

AutoGluon enables easy-to-use and easy-to-extend AutoML with a focus on automated stack ensembling, deep learning, and real-world applications spanning text, image, and tabular data. 

AutoGluon enables you to:

- Quickly prototype deep learning and classical ML solutions for your raw data with a few lines of code.

- Automatically utilize state-of-the-art techniques (where appropriate) without expert knowledge.

- Leverage automatic hyperparameter tuning, model selection/ensembling, architecture search, and data processing.

- Easily improve/tune your models and data pipelines or customize AutoGluon for your use-case.


### [AutoKeras](https://github.com/keras-team/autokeras)

AutoKeras is an AutoML system based on Keras developed by DATA Lab at Texas A&M University. 

The goal of AutoKeras is to make machine learning accessible to everyone.


### [Auto-ViML](https://github.com/AutoViML/Auto_ViML)

Auto_ViML was designed for building High Performance Interpretable Models with the fewest variables.

[Why AutoML Is An Essential New Tool For Data Scientists](https://towardsdatascience.com/why-automl-is-an-essential-new-tool-for-data-scientists-2d9ab4e25e46)

The "V" in Auto_ViML stands for Variable because it tries multiple models with multiple features to find you the best performing model for your dataset. 

The "i" in Auto_ViML stands for "interpretable" since Auto_ViML selects the least number of features necessary to build a simpler, more interpretable model. 

In most cases, Auto_ViML builds models with 20-99% fewer features than a similar performing model with all included features.

Auto_ViML is every Data Scientist's model assistant that:

1. Helps you with data cleaning

2. Assists you with variable classification

3. Performs feature reduction automatically

4. Produces model performance results as graphs automatically

5. Handles text, date-time, structs (lists, dictionaries), numeric, boolean, factor and categorical variables all in one model using one straight process.

6. Allows you to use the featuretools library to do Feature Engineering.


### [AutoViz](https://github.com/AutoViML/AutoViz)

Automatically Visualize any dataset of any size with a single line of code.

[AutoViz: A New Tool for Automated Visualization](https://towardsdatascience.com/autoviz-a-new-tool-for-automated-visualization-ec9c1744a6ad)

AutoViz performs automatic visualization of any dataset with one line. 

Give any input file (CSV, txt, or json) and AutoViz will visualize it.


### [MediaPipe](https://github.com/google/mediapipe)

Live ML anywhere. 

MediaPipe offers cross-platform, customizable ML solutions for live and streaming media.

- Face Detectiom
- Face Mesh
- Pose
- Object Detection
- Box Tracking
- Objectron


### [Ray](https://github.com/ray-project/ray)

An open source framework that provides a simple, universal API for building distributed applications. 

Ray is packaged with `RLlib` which is a scalable reinforcement learning library and `Tune` which is a scalable hyperparameter tuning library.


### [FLAML](https://github.com/microsoft/FLAML)

FLAML is a fast, lightweight Python library that finds accurate machine learning models automatically which frees users from selecting learners and hyperparameters for each learner.

The simple and lightweight design makes it easy to extend such as adding customized learners or metrics. 

FLAML is powered by a new, cost-effective hyperparameter optimization and learner selection method invented by Microsoft Research. 

FLAML leverages the structure of the search space to choose a search order optimized for both cost and error.


### [SageMaker](https://aws.amazon.com/sagemaker/?nc2=h_a1)

SageMaker is a machine learning environment that simplifies the work of an ML developer by providing tools for extra fast model building and deployment.

In 2021, Amazon launched SageMaker Studio, the first integrated IDE for machine learning that provides a web interface to monitor all possible aspects of the life cycle of an ML model, basicslly Jupyter on steroids. 

SageMaker is closely integrated into the AWS cloud and it also offers data labeling software and other features.


---------


## Time Series Libraries

- PyCaret

- AutoGluon
- AutoKeras
- AutoTS
- Darts
- mlforecast

- TimeSynth
- TimeGAN
- Gretel.ai

