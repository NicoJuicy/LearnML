# Frequently Asked Questions

## Do I need a Master’s Degree?

[Do you need a Master’s Degree in Data Science?](https://towardsdatascience.com/do-you-need-a-masters-degree-in-data-science-you-are-asking-the-wrong-question-8c83dec8bf1b?source=rss----7f60cf5620c9---4)

If you are going to spend the time to study AI/ML then you might as well invest in an online degree which will greatly increase your career opportunities (and a requirement for most all AI/ML engineer positions).


## Recommended Tutorials and Books

See "Tutorials and Books" in [How to Learn Machine Learning](https://aicoder.medium.com/how-to-learn-machine-learning-4ba736338a56)


## How to ask an AI/ML question?

Briefly describe the following (1-2 sentences per item):

1. Give some description of your background and experience. 
2. Describe the problem. 
3. Describe the dataset in detail and be willing to share your dataset.
4. Describe any data preparation and feature engineering steps that you have done.
5. Describe the models that you have tried. 
6. Favor text and tables over plots and graphs.
7. Avoid asking users to help debug your code. 

See [How to ask an AI/ML question](https://aicoder.medium.com/how-to-ask-an-ai-ml-question-6cfddaa75bc9)


## How to choose a performance metric?

See [Machine Learning Performance Metrics](./ml/performance_metrics.md) 


## How to Choose an ML Algorithm?

See [How to Choose an ML Model](./getting_started.md) and [Understand Machine Learning Algorithms](./ml/getting_started.md)


## Should I start learning ML by coding an algorithm from scratch?

See [How to Learn Machine Learning](https://aicoder.medium.com/how-to-learn-machine-learning-4ba736338a56)


## Is image channels first or last?

A huge gotcha with both PyTorch and Keras. Actually, you need to sometimes need to to watch out when running code between OS (NHWC vs NCHW). I spent a long time tracking down an obscure error message between Linux and macOS that turned out to be the memory format.

[PyTorch Channels Last Memory Format Performance Optimization on CPU Path](https://gist.github.com/mingfeima/595f63e5dd2ac6f87fdb47df4ffe4772)

[Change Image format from NHWC to NCHW for Pytorch](https://stackoverflow.com/questions/51881481/change-image-format-from-nhwc-to-nchw-for-pytorch)

[A Gentle Introduction to Channels-First and Channels-Last Image Formats](https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/)



## How to share your work?

Your are always welcome to share your work in the following Discord AI forum channels of the SHARE KNOWLEDGE Section: #share-your-projects, #share-datasets, and #our-accomplishments. 


## How to choose a Cloud Platform?

[Comparison of Basic Deep Learning Cloud Platforms](https://aicoder.medium.com/comparison-of-basic-deep-learning-cloud-platforms-9a4b69f44a46)


## Medical Images

Small and imbalanced datasets are common in medical applications. However, it is still considered an open research problem in CS. Thus, there is not standard “recipe” for data prep. Just some heuristics that people have come up with. So u will need to do some research to justify your final choice of data prep techniques, especially for medical datasets. At least one of the articles discusses X-ray images which may have some references that are helpful (not sure). I would also try searching on arxiv.org for “Survey” articles that would list some peer-reviewed journal articles on the type of images that u are working with.

Resampling is just one approach to balance a dataset but it is an advanced concept, so u need to have a thorough understanding of the core ML concepts. Otherwise, your results will be suspect. I have some notes on “Dataset Issues” that may help get u started. However, the approach is different for image datasets.


