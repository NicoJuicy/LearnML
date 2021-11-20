# AutoML Tools

Automated Machine Learning (AutoML) is an emerging field in which the process of building machine learning models to model data is automated.

Here is a list of AutoML tools that I have found to be helpful to an AI/ML engineer. 


### [FLAML](https://github.com/microsoft/FLAML)

FLAML is a fast, lightweight Python library that finds accurate machine learning models automatically which frees users from selecting learners and hyperparameters for each learner.

The simple and lightweight design makes it easy to extend such as adding customized learners or metrics. 

FLAML is powered by a new, cost-effective hyperparameter optimization and learner selection method invented by Microsoft Research. 

FLAML leverages the structure of the search space to choose a search order optimized for both cost and error.


### [PyCaret](https://github.com/pycaret/pycaret)

PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive.

In comparison with the other open-source machine learning libraries, PyCaret is an alternate low-code library that can be used to replace hundreds of lines of code with few words only. This makes experiments exponentially fast and efficient. PyCaret is essentially a Python wrapper around several machine learning libraries and frameworks such as scikit-learn, XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray, and many more.


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


### [wandb](https://docs.wandb.ai/)

Weights and biases (W&B/wandb) lets you track, compare, visualize, and optimize machine learning experiments with just a few lines of code. wandb also lets you track your datasets. 

Using W&B's lightweight tools, you can quickly track experiments, version and iterate on datasets, evaluate model performance, reproduce models, visualize results and spot regressions, and share findings with colleagues. 


### [MLFlow](https://github.com/mlflow/mlflow)

[Tutorial](https://www.mlflow.org/docs/latest/tutorials-and-examples/tutorial.html)

Similar to W&B, MLFlow provides functionality for logging code, models, and datasets on which your model has been trained. 

MLflow is a platform to streamline machine learning development including tracking experiments, packaging code into reproducible runs, and deploying models. 

MLflow offers a set of lightweight APIs that can be used with any existing machine learning application or library (TensorFlow, PyTorch, XGBoost, etc.) wherever you currently run ML code (such as notebooks, standalone applications, or the cloud). 


### [auto-sklearn](https://github.com/automl/auto-sklearn)

auto-sklearn is an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator.


### [AutoGluon](https://github.com/awslabs/autogluon)

AutoGluon automates machine learning tasks enabling you to easily achieve strong predictive performance in your applications. With just a few lines of code, you can train and deploy high-accuracy machine learning and deep learning models on text, image, and tabular data.


### [AutoKeras](https://github.com/keras-team/autokeras)

AutoKeras is an AutoML system based on Keras developed by DATA Lab at Texas A&M University. 

The goal of AutoKeras is to make machine learning accessible to everyone.


### [Auto-ViML](https://github.com/AutoViML/Auto_ViML)

Auto_ViML was designed for building High Performance Interpretable Models with the fewest variables. The "V" in Auto_ViML stands for Variable because it tries multiple models with multiple features to find you the best performing model for your dataset. The "i" in Auto_ViML stands for "interpretable" since Auto_ViML selects the least number of features necessary to build a simpler, more interpretable model. In most cases, Auto_ViML builds models with 20-99% fewer features than a similar performing model with all included features (this is based on my trials. Your experience may vary).

Auto_ViML is every Data Scientist's model assistant that:

1. Helps you with data cleaning

2. Assists you with variable classification

3. Performs feature reduction automatically

4. Produces model performance results as graphs automatically

5. Handles text, date-time, structs (lists, dictionaries), numeric, boolean, factor and categorical variables all in one model using one straight process.

6. Allows you to use the featuretools library to do Feature Engineering.


### [AutoViz](https://github.com/AutoViML/AutoViz)

Automatically Visualize any dataset, any size with a single line of code.

AutoViz performs automatic visualization of any dataset with one line. Give any input file (CSV, txt or json) and AutoViz will visualize it.


### [MediaPipe](https://github.com/google/mediapipe)

Live ML anywhere

MediaPipe offers cross-platform, customizable ML solutions for live and streaming media.

- Face Detectiom
- Face Mesh
- Pose
- Object Detection
- Box Tracking
- Objectron


