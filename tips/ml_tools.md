# Machine Learning Tools

Here is a list of ML tools that I have found to be helpful to an AI/ML engineer.

[GitHub Lists](https://github.com/codecypher?tab=stars)

<!-- MarkdownTOC levels=1,2 -->

- How to choose an ML framework?
- Keras Tutorials
- Data Exploration Tools
- Feature Engineering Tools
- ML Libraries
- Computer Vision
- Natural Language Programming
- Python Libraries
- Plots and Graphs
- Deep Learning Tools
- Linux Utilities
- Time Series
- Audio
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

- Orange
- DataPrep
- Bamboolib
- TensorFlow Data Validation
- Great Expectations

NOTE: It is best to install the Orange native executable on your local machine rather than install using anaconda and/or pip.

### Tutorials

[Orange Docs](https://orangedatamining.com/docs/)

[A Great Python Library: Great Expectations](https://towardsdatascience.com/a-great-python-library-great-expectations-6ac6d6fe822e)



## Feature Engineering Tools

There are many tools that will help you in automating the entire feature engineering process and producing a large pool of features in a short period of time for both classification and regression tasks.

- Feature-engine
- Featuretools
- AutoFeat

### Tutorials

[The Only Web Scraping Tool you need for Data Science](https://medium.com/nerd-for-tech/the-only-web-scraping-tool-you-need-for-data-science-f388e2afa187)



## ML Libraries

- Kedro
- ONNX
- openai/gym
- PyMC
- Snap ML
- Streamlit

### models

The models repo is also called “TensorFlow Modul Garden” which organises machine learning models and has them implemented using TensorFlow with examples. The models could be from TensorFlow officially, from some famous research projects or the community. They can be very helpful to save our time when we want to use any machine learning models in TensorFlow.

### Snap ML

[Snap ML](https://www.zurich.ibm.com/snapml/)

Snap ML is a library that provides high-speed training of popular machine learning models on modern CPU/GPU computing systems

[This Library is 30 Times Faster Than Scikit-Learn](https://medium.com/@irfanalghani11/this-library-is-30-times-faster-than-scikit-learn-206d1818d76f)

[IBM Snap ML Examples](https://github.com/IBM/snapml-examples)


### Tutorials

[Introduction to OpenCV](https://www.geeksforgeeks.org/introduction-to-opencv/)

[OpenCV Python Tutorial](https://www.geeksforgeeks.org/opencv-python-tutorial/)

[Kedro — A Python Framework for Reproducible Data Science Project](https://towardsdatascience.com/kedro-a-python-framework-for-reproducible-data-science-project-4d44977d4f04)

[How to start contributing to open-source projects](https://towardsdatascience.com/how-to-start-contributing-to-open-source-projects-41fcfb654b2e)

[A Gentle Introduction to Bayesian Belief Networks](https://machinelearningmastery.com/introduction-to-bayesian-belief-networks/)

[Building DAGs with Python](https://mungingdata.com/python/dag-directed-acyclic-graph-networkx/)

[bnlearn](https://github.com/erdogant/bnlearn)



## Computer Vision

- ageitgey/face_recognition

### [OpenCV](https://docs.opencv.org/master/d9/df8/tutorial_root.html)

OpenCV is a huge open-source library for computer vision, machine learning, and image processing. 

OpenCV supports a wide variety of programming languages like Python, C++, Java, etc. 

OpenCV can process images and videos to identify objects, faces, or eve handwriting.

### openpilot

openpilot repo is an open-source driver assistance system. It performs the functions of Adaptive Cruise Control (ACC), Automated Lane Centering (ALC), Forward Collision Warning (FCW) and Lane Departure Warning (LDW) for a growing variety of supported car makes, models and model years. However, you need to buy their product and install it on your car. It is not completely DIY but reduces the efforts.



## Natural Language Programming

- Polyglot
- SpaCy
- GenSim
- Pattern
- clean-text

### [NLTK](https://www.nltk.org)

[NLTK Book](https://www.nltk.org/book/)

Natural language toolkit (NLTK) is by far the most popular platform for building NLP related projects.

NLTK is also an open-source library and is available for almost every kind of operating system. 



## Python Libraries

- dateutil
- Pipreqs
- Poetry
- tqdm
- urllib3

- The Algorithms - Python
- vinta/awesome-python
- josephmisiti/awesome-machine-learning


### Jupyterlab

[JupyterLab](https://github.com/jupyterlab/jupyterlab) is the next-generation user interface for Project Jupyter offering all the familiar building blocks of the classic Jupyter Notebook (notebook, terminal, text editor, file browser, rich outputs, etc.) in a flexible and powerful user interface. JupyterLab will eventually replace the classic Jupyter Notebook.

Jupyterlab has an updated UI/UX with a tab interface for working with multiple files and notebooks.

Since Jupyter is really a web server application, it runs much better on a remote server. 

I currently have Jupyterlab installed and running as a Docker container on a VM droplet which runs much better than on my local machine. The only issue is that my VM only has 2GB memory. However, I have had great success so far using Jupyterlab and Modin with notebooks that I am unable to run on my local machine with 32GB memory (out of memory issues) without any performance issues.

If you do not have cloud server of your own, a nice alternative is [Deepnote](https://deepnote.com). The free tier does not offer GPU access but it does offer a shared VM with 24GB of memory running a custom version of Jupyterlab which I have found more useful than Google Colab Pro. It is definitely worth a try. 

### Modin

[Modin](https://github.com/modin-project/modin) is a drop-in replacement for pandas. 

While pandas is single-threaded, Modin lets you speed up your workflows by scaling pandas so it uses all of your cores. 

Modin works especially well on larger datasets where pandas becomes painfully slow or runs out of memory.

Using modin is as simple as replacing the pandas import:

```py
  # import pandas as pd
  import modin.pandas as pd
```

I have a sample [notebook](../python/book_recommender_knn.py) that demonstrates using modin. 

Since Modin is still under development, I do experience occasional warning/error messages but everything seems to be working. However, the developers seem to be quick to answer questions and provide assistance in troubleshooting issues. Highly recommend trying it out. 


### Pickle

Pickle can be used to save and load the python classes or PyTorch models for reuse.

### PySpark

[Getting Started](https://spark.apache.org/docs/latest/api/python/getting_started/index.html)

PySpark is an interface for Apache Spark in Python. It not only allows you to write Spark applications using Python APIs, but also provides the PySpark shell for interactively analyzing your data in a distributed environment. PySpark supports most of Spark’s features such as Spark SQL, DataFrame, Streaming, MLlib (Machine Learning) and Spark Core.

### Debugging Tools

- heartrate
- Loguru
- snoop



## Plots and Graphs

[Plots](./plots.md)



## Deep Learning Tools

- MXNet

### [H5py](https://docs.h5py.org/en/stable/quick.html)

H5py can be used to store all the intermediate loss values in a dictionary mapped to appropriate key which can be loaded to be reused as a python code.

### Tutorials

[Are You Still Using Virtualenv for Managing Dependencies in Python Projects?](https://towardsdatascience.com/poetry-to-complement-virtualenv-44088cc78fd1)

[3 Tools to Track and Visualize the Execution of Your Python Code](https://www.kdnuggets.com/2021/12/3-tools-track-visualize-execution-python-code.html)



## Linux Utilities

- awk
- tmux
- screen

### awk

awk is a pattern scanning and text processing language which is also considered a programming language specifically designed for processing text.

### [Lucidchart](https://www.lucidchart.com/pages/)

Lucidchart is a diagraming tool that also has shared space for collaboration and the ability to make notes next to diagrams.

### [Screen](https://linuxize.com/post/how-to-use-linux-screen/)

Screen is a GNU linux utility that lets you launch and use multiple shell sessions from a single ssh session. The process started with screen can be detached from session and then reattached at a later time. So your experiments can be run in the background, without the need to worry about session closing, or terminal crashing.

### tmux

tmux is a terminal multiplexer that allows you to access a tmux terminal using multiple virtual terminals.

tmux takes advantage of a client-server model which allows you to attach terminals to a tmux session which means: 

- You can run several terminals at once concurrently off a single tmux session without spawning any new terminal sessions.

- Sudden disconnects from a cloud server running tmux will not kill the processes running inside the tmux session.

tmux also includes a window-pane mentality which means you can run more than one terminal on a single screen.

### httpie

HTTPie is a command-line HTTP client. Its goal is to make CLI interaction with web services as human-friendly as possible. HTTPie is designed for testing, debugging, and generally interacting with APIs & HTTP servers. 

The http and https commands allow for creating and sending arbitrary HTTP requests using a simple and natural syntax and provide formatted and colourized output.

### rich

rich makes it easy to add colour and style to terminal output. It can also render pretty tables, progress bars, markdown, syntax highlighted source code, tracebacks, and more — out of the box.

### Tutorials

[How To Install And Use tmux On Ubuntu 12.10](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-tmux-on-ubuntu-12-10--2)

[10 Practical Uses of AWK Command for Text Processing](https://betterprogramming.pub/10-practical-use-of-awk-command-in-linux-unix-26fbd92f1112)

[Display Rich Text In The Console Using rich](https://towardsdatascience.com/get-rich-using-python-af66176ece8f?source=linkShare-d5796c2c39d5-1641842633)



## Time Series

- statsmodels
- stumpy
- AutoTS
- Darts
- TsFresh



## Audio

- [AssemblyAI](https://www.assemblyai.com/)



## References

[All Top Python Libraries for Data Science Explained](https://towardsdatascience.com/all-top-python-libraries-for-data-science-explained-with-code-40f64b363663)

[26 GitHub Repositories To Inspire Your Next Data Science Project](https://towardsdatascience.com/26-github-repositories-to-inspire-your-next-data-science-project-3023c24f4c3c)

[7 Amazing Python Libraries For Natural Language Processing](https://towardsdatascience.com/7-amazing-python-libraries-for-natural-language-processing-50ca6f9f5f11)

[4 Amazing Python Libraries That You Should Try Right Now](https://towardsdatascience.com/4-amazing-python-libraries-that-you-should-try-right-now-872df6f1c93)

[Colaboratory FAQ](https://research.google.com/colaboratory/faq.html#resource-limits)

[Tools for Efficient Deep Learning](https://towardsdatascience.com/tools-for-efficient-deep-learning-c9585122ded0)

