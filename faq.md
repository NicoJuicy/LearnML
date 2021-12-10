# Frequently Asked Questions

<!-- MarkdownTOC -->

- Do I need a Master’s Degree?
- Recommended Tutorials and Books
- How to ask an AI/ML question?
- How to choose a performance metric?
- How to Choose an ML Algorithm?
- Should I start learning ML by coding an algorithm from scratch?
- Is image channels first or last?
- How to share your work?
- How to choose a Cloud Platform?
- Common Questions on Encoding
    - What if I have hundreds of categories?
    - What encoding technique is the best?
    - What if I have a mixture of numeric and categorical data?
- Common Questions on Normalization
    - Which Scaling Technique is Best?
    - Should I Normalize or Standardize?
    - Should I Standardize then Normalize?
    - How Do I Handle Out-of-Bounds Values?
- Medical Images
- Why are Robots not more common?

<!-- /MarkdownTOC -->


## Do I need a Master’s Degree?

If you are going to spend the time to study AI/ML then you might as well invest in an online degree which will greatly increase your career opportunities (and a requirement for most all AI/ML engineer positions).

[Do you need a Master’s Degree in Data Science?](https://towardsdatascience.com/do-you-need-a-masters-degree-in-data-science-you-are-asking-the-wrong-question-8c83dec8bf1b?source=rss----7f60cf5620c9---4)

[Do You Have the Degree it Takes to Get Hired as a Data Scientist?](https://medium.com/@ODSC/do-you-have-the-degree-it-takes-to-get-hired-as-a-data-scientist-e87e2d6c0c87)

[Why Machine Learning Engineers are Replacing Data Scientists](https://www.kdnuggets.com/2021/11/why-machine-learning-engineers-are-replacing-data-scientists.html)


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

The best approach would be to find several job postings that look interesting to you and see what skills and tools they require.  


## Is image channels first or last?

A huge gotcha with both PyTorch and Keras. Actually, you need to sometimes need to to watch out when running code between OS (NHWC vs NCHW). I spent a long time tracking down an obscure error message between Linux and macOS that turned out to be the memory format.

[PyTorch Channels Last Memory Format Performance Optimization on CPU Path](https://gist.github.com/mingfeima/595f63e5dd2ac6f87fdb47df4ffe4772)

[Change Image format from NHWC to NCHW for Pytorch](https://stackoverflow.com/questions/51881481/change-image-format-from-nhwc-to-nchw-for-pytorch)

[A Gentle Introduction to Channels-First and Channels-Last Image Formats](https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/)



## How to share your work?

Your are always welcome to share your work in the following Discord AI forum channels of the SHARE KNOWLEDGE Section: #share-your-projects, #share-datasets, and #our-accomplishments. 


## How to choose a Cloud Platform?

[Comparison of Basic Deep Learning Cloud Platforms](https://aicoder.medium.com/comparison-of-basic-deep-learning-cloud-platforms-9a4b69f44a46)



## Common Questions on Encoding

This section lists some common questions and answers when encoding categorical data.

### What if I have hundreds of categories?

What if I concatenate many one-hot encoded vectors to create a many-thousand-element input vector?

You can use a one-hot encoding up to thousands and tens of thousands of categories. Having large vectors as input sounds intimidating but the models can usually handle it.

### What encoding technique is the best?

This is impossible to answer. The best approach would be to test each technique on your dataset with your chosen model and discover what works best.

### What if I have a mixture of numeric and categorical data?

What if I have a mixture of categorical and ordinal data?

You will need to prepare or encode each variable (column) in your dataset separately then concatenate all of the prepared variables back together into a single array for fitting or evaluating the model.

Alternately, you can use the `ColumnTransformer` to conditionally apply different data transforms to different input variables.


## Common Questions on Normalization

This section lists some common questions and answers when scaling numerical data.

### Which Scaling Technique is Best?

This is impossible to answer. The best approach would be to evaluate models on data prepared with each transform and use the transform or combination of transforms that result in the best performance for your data set and model.

### Should I Normalize or Standardize?

Whether input variables require scaling depends on the specifics of your problem and of each variable.

If the distribution of the values is normal, it should be standardized. Otherwise, the data should be normalized.

The data should be normalized whether the range of quantity values is large (10s, 100s, ...) or small (0.01, 0.0001, ...).

If the values are small (near 0-1) and the distribution is limited (standard deviation near 1) you might be able to get away with no scaling of the data.

Predictive modeling problems can be complex and it may not be clear how to best scale input data.

If in doubt, normalize the input sequence. If you have the resources, explore modeling with the raw data, standardized data, and normalized data and see if there is a difference in the performance of the resulting model.

### Should I Standardize then Normalize?

Standardization can give values that are both positive and negative centered around zero.

It may be desirable to normalize data after it has been standardized.

Standardize then Normalize may be a good approach if you have a mixture of standardized and normalized variables and would like all input variables to have the same minimum and maximum values as input for a given algorithm such as an algorithm that calculates distance measures.

### How Do I Handle Out-of-Bounds Values?

You may normalize your data by calculating the minimum and maximum on the training data.

Later, you may have new data with values smaller or larger than the minimum or maximum respectively.

One simple approach to handling this may be to check for out-of-bound values and change their values to the known minimum or maximum prior to scaling. Alternately, you can estimate the minimum and maximum values used in the normalization manually based on domain knowledge.


## Medical Images

Small and imbalanced datasets are common in medical applications. However, it is still considered an open research problem in CS. Thus, there is not standard “recipe” for data prep. Just some heuristics that people have come up with. So u will need to do some research to justify your final choice of data prep techniques, especially for medical datasets. 

At least one of the articles discusses X-ray images which may have some references that are helpful (not sure). I would also try searching on arxiv.org for “Survey” articles that would list some peer-reviewed journal articles on the type of images that u are working with.

Resampling is just one approach to balance a dataset but it is an advanced concept, so u need to have a thorough understanding of the core ML concepts. Otherwise, your results will be suspect. 

I have some notes on “Dataset Issues” that may help get u started. However, the approach is different for image datasets.


## How to Develop a Chatbot?

Chatbots are better to use pretrained model and software. You can take a look at Moodle and Rasa which are popular. There is also an example using NLTK that claims to be somewhat accurate. 

[How to Create a Moodle Chatbot Without Any Coding?](https://chatbotsjournal.com/how-to-create-a-moodle-chatbot-without-any-coding-3d08f95d94df)

[Building a Chatbot with Rasa](https://towardsdatascience.com/building-a-chatbot-with-rasa-3f03ecc5b324)

[Python Chatbot Project – Learn to build your first chatbot using NLTK and Keras](https://data-flair.training/blogs/python-chatbot-project/)


## Why are Robots not more common?

Robot soccer is one type of classic robotics toy problem, often involving multiagent reinforcement learning (MARL). 

In robotics, there are a lot of technical issues (mainly safety related) involved besides the biomechanics of walking. 

There has been limited success for certain application areas such as the robot dog and of course robotic kits for arduino and raspberry pi (for hobbyists) but more practical applications still seem to be more elusive. 

In general, it costs a lot of money for R&D in robotics and it takes a long time for ROI. For example, openai recently disbanded its robotics division dues to lack of data and capital. Perhaps more interesting is the lack of a proper dataset which is needed for real-world robotics applications in the wild.


[OpenAI disbands its robotics research team](https://venturebeat.com/2021/07/16/openai-disbands-its-robotics-research-team/)

[Boston Dynamics now sells a robot dog to the public, starting at $74,500](https://arstechnica.com/gadgets/2020/06/boston-dynamics-robot-dog-can-be-yours-for-the-low-low-price-of-74500/)



