# Natural Language Processing (NLP)

<!-- MarkdownTOC -->

- Overview
- NLP Cheatsheet
- Data Mining
- Pattern Recognition
- Challenges in Natural Language Processing
- Common Tasks
- NLP Workflow
- Lemmatization and Stemming
- Keyword extraction
- Named Entity Recognition \(NER\)
- Topic Modeling
- Summarization
- Sentiment Analysis
  - Sentiment Analysis using Python
  - Sentiment Analysis using AutoML
- Applications
- References

<!-- /MarkdownTOC -->


## Overview

Natural Language Processing (NLP) is a category of ML concerned with the analysis and building of intelligent systems that can function in languages that humans speak. 

Processing of language is needed when a system wants to work based on input from a user in the form of text or speech and the user is adding input in regular use English.

- Natural Language Understanding (NLU): The understanding phase of the processing is responsible for mapping the input that is given in natural language to a beneficial representation. It also analyzes different aspects of the input language that is given to the program.

- Natural Language Generation (NLG): The generation phase of the processing is used in creating Natural Languages from the first phase. 

Generation starts with Text Planning, which is the extraction of relevant content from the base of knowledge. 

Next, the Sentence Planning phase is where required words that will form the sentence are chosen. 

Finally, Text Realization is the final creation of the sentence structure.

Researchers are now using NLP to understand the meaning of large sets of documents from an analysis perspective which means that they can understand the different contexts and nuances of what phrases mean.

Natural language toolkit (NLTK) is the defacto standard for building NLP  projects.



## NLP Cheatsheet

This article is a checklist for the  exploration needed to develop an NLP model that performs well. 

[NLP Cheatsheet](https://medium.com/javarevisited/nlp-cheatsheet-2b19ebcc5d2e)



## Data Mining

Data mining is the process of analyzing data by searching for patterns to turn the data into information and better decisions. 

Data mining is algorithm-based and finds patterns in large collections of data. 

Data mining is also important because it presents a potentially more efficient and thorough way of interpreting data.

## Pattern Recognition

Pattern recognition is a branch of ML that is focused on categorizing information or finding anomalies in data. For example, facial pattern recognition might be used to determine the age and gender of a person in a photo. 

Pattern recognition tends to be based on probability, so there is a chance that it does not accurately categorize information. 

Pattern recognition is also typically controlled by an algorithm which means that the computer will continue to make guesses until it finds a pattern that matches what we know is true or until the probability of any other pattern remaining is too small to be considered.


## Challenges in Natural Language Processing

- **Lexical Ambiguity:** This is the first level of ambiguity that occurs generally in words alone. For instance, when a code is given a word like ‘board’ it would not know whether to take it as a noun or a verb. This causes ambiguity in the processing of this piece of code.

- **Syntax Level Ambiguity:** This is another type of ambiguity that has more to do with the way phrases sound in comparison to how the machine perceives it. For instance, a sentence like, ‘He raised the scuttle with a blue cap’. This could mean one of two things. Either he raised a scuttle with the help of a blue cap, or he raised a scuttle that had a red cap.

- **Referential Ambiguity:** References made using pronouns constitute referential ambiguity. For instance, two girls are running on the track. Suddenly, she says, ‘I am exhausted’. It is not possible for the program to interpret, who out of the two girls is tired.


## Common Tasks

Some NLP use cases and tasks are:

- **Sentiment Analysis:** to understand the sentiment (negative, positive, neutral) a certain document/text holds.

Example: social media posts about Climate Change.

- **Topic Modeling:** to draw clusters or organize the data based on the topics (goal is to learn these topics) it contains.

Example: an insurance company wants to identify fraudulent claims by classifying causes into main labels and then further analyze the ones containing suspicious content/topics.

- **Text Generation:** to create new data (textual mainly) based on previous examples from the same domain. Example: chatbots, quotes, email replies, etc.

Example: chatbots, quotes , email replies, …, etc.

- **Machine Translation:** to automatically convert from one language to another.

Example: English to German.



## NLP Workflow

[Text Augmentation in a few lines of Python Code](https://towardsdatascience.com/text-augmentation-in-few-lines-of-python-code-cdd10cf3cf84)

[Hands on Implementation of Basic NLP Techniques: NLTK or spaCy](https://towardsdatascience.com/hands-on-implementation-of-basic-nlp-techniques-nltk-or-spacy-687099e02816)

In Data Science (and NLP) there is a workflow or pipeline that can be described as follows:

1. Define the Question which you want to answer out of your data.

Usually this question is given to you as the problem but sometimes it is your job to articulate it. 

2. Get and collect your Data. 

If your problem is in the domain of movie reviews your data would be viewers posted reviews along with the ratings. ￼

It is critical that your data is in the same domain as your question/problem and comprehensive, most of times the data is provided or at least the resources that you should be looking at to obtain it.

3. Clean your data. ￼

Almost 90% of the time the data you have is raw, unclean, contains missing fields/outliers/misspellings and so on. 

4. Perform Exploratory Analysis of your Data (EDA). 

EDA is one of the most important steps in any Data Science or NLP task. 

After you have brought your data into a clean ready-to-use state, you want to explore it such that you understand more of its nature and content. 

Your analysis should keep the problem’s question in mind and your job is to try to connect the dots as this step might yield in finding useful correlations/outliers and trends in your data. 

5. Run the NLP technique which suits best your problem which means deciding whether your problem requires sentiment analysis or topic modelling or any other advanced technique which deals with textual data. 

With some practice and experience you would be able to quickly identify the best NLP approach to solve a certain problem. Keep in mind that you can also perform multiple techniques on a single problem in order to be able to draw conclusions and to obtain insights that will answer the main question in step 1. ￼

Deciding on an approach/technique usually means choosing the suitable model or library/package to perform the task. 

6. Obtain knowledge and insights. 

In this step, you need to make use of your communication and representation skills as a data scientist. 



## Lemmatization and Stemming

Stemming and lemmatization are probably the first two steps to build an NLP project — you often use one of the two. 

- Stemming: Stemming is a collection of algorithms that work by clipping off the end of the beginning of the word to reach its infinitive form. 

  These algorithms find the common prefixes and suffixes of the language being analyzed. 

Clipping off the words can lead to the correct infinitive form, but that’s not always the case. 

There are many algorithms to perform stemming; the most common one used in English is the Porter stemmer which contains 5 phases that work sequentially to obtain the word’s root.

- Lemmatization: To overcome the flaws of stemming, lemmatization algorithms were designed. 

In these types of algorithms, some linguistic and grammar knowledge needs to be fed to the algorithm to make better decisions when extracting a word’s infinitive form. 

For lemmatization algorithms to perform accurately, they need to extract the correct lemma of each word. Thuz, they often require a _dictionary_ of the language to be able to categorize each word correctly.


## Keyword extraction

Keyword extraction (keyword detection or keyword analysis) is an NLP technique used for text analysis. 

This main purpose of keyword extraction (KE) is to automatically extract the most frequent words and expressions from the body of a text. 

KE is often used as a first step to summarize the main ideas of a text and to deliver the key ideas presented in the text.


## Named Entity Recognition (NER)

Similar to stemming and lemmatization, named entity recognition (NER) is a technique used to extract entities from a body of a text used to identify basic concepts within the text such as people's names, places, dates, etc.

NER algorithm has mainly two steps. 

  1. It needs to detect an entity in the text
  2. It categorizes the text into one category. 

The performance of NER depends heavily on the training data used to develop the model. The more relevant the training data to the actual data, the more accurate the results will
be.

## Topic Modeling

You can use keyword extraction techniques to narrow down a large body of text to a handful of main keywords and ideas. Then, you can extract the main topic of the text.

Another, more advanced technique to identify a the topic of text is topic modeling which is built upon unsupervised machine learning that does not require a labeled data for training.


## Summarization

One of the useful and promising applications of NLP is text summarization which is reducing a large body of text into a smaller chuck containing the text's main message. 

This technique is often used in long news articles and to summarize research papers.

Text summarization is an advanced technique that uses other techniques that we just mentioned to establish its goals such as topic modeling and keyword extraction. 

Summarization is accomplished in two steps: extract and abstract.


## Sentiment Analysis

The most famous and most commonly used NLP technique is sentiment analysis (SA). 

The core function of SA is to extract the sentiment behind a body of text by analyzing the containing words.

The technique's most simple results lay on a trinary scale: negative, positive, and neutral. 

The SA algorithm can be more complex and advanced; however, the results will be numeric in this case. 

If the result is a negative number, the sentiment behind the text has a negative tone to it, and if it is positive then some positivity is present in the text.

### Sentiment Analysis using Python

[Best Practices for Text Classification with Deep Learning](https://machinelearningmastery.com/best-practices-document-classification-deep-learning/)

[How to Develop a Deep Learning Bag-of-Words Model for Sentiment Analysis (Text Classification)](https://machinelearningmastery.com/deep-learning-bag-of-words-model-sentiment-analysis/)

[Deep Convolutional Neural Network for Sentiment Analysis (Text Classification)](https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/)

[How to Use Word Embedding Layers for Deep Learning with Keras](https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/)

[Keras Code Examples](https://keras.io/examples/)


----------


[Twitter Sentiment Analysis using NLTK and Python](https://towardsdatascience.com/twitter-sentiment-analysis-classification-using-nltk-python-fa912578614c)

[Classifying Tweets for Sentiment Analysis: NLP in Python for Beginners](https://medium.com/vickdata/detecting-hate-speech-in-tweets-natural-language-processing-in-python-for-beginners-4e591952223)

[Tweet Classification and Clustering in Python](https://medium.com/swlh/tweets-classification-and-clustering-in-python-b107be1ba7c7)

[Identifying Tweet Sentiment in Python](https://towardsdatascience.com/identifying-tweet-sentiment-in-python-7c37162c186b)


### Sentiment Analysis using AutoML

[A Gentle Introduction to PyCaret for Machine Learning](https://machinelearningmastery.com/pycaret-for-machine-learning/)

[NLP Text-Classification in Python: PyCaret Approach vs The Traditional Approach](https://towardsdatascience.com/nlp-classification-in-python-pycaret-approach-vs-the-traditional-approach-602d38d29f06)

[Natural Language Processing Tutorial (NLP101) - Level Beginner](http://www.pycaret.org/tutorials/html/NLP101.html)


----------


[Complete Guide to Perform Classification of Tweets with SpaCy](https://towardsdatascience.com/complete-guide-to-perform-classification-of-tweets-with-spacy-e550ee92ca79)

[Sentiment Analysis of Tweets using BERT](https://thinkingneuron.com/sentiment-analysis-of-tweets-using-bert/)

[How to use SHAP with PyCaret](https://astrobenhart.medium.com/how-to-use-shap-with-pycaret-dc9a31278621)

[Fine-Tuning BERT for Tweets Classification with HuggingFace](https://www.kdnuggets.com/2022/01/finetuning-bert-tweets-classification-ft-hugging-face.html)



## Applications

- Document Clustering

The general idea with document clustering is to assign each document a vector representing the topics discussed. 

- Sentiment Analysis: Naive Bayes, gradient boosting, and random forest

- Keyword Extraction: Named Entity Recognition (NER) using SpaCy, Rapid Automatic Keyword Extraction (RAKE) using ntlk-rake

- Text Summarization: TextRank (similar to PageRank) using PyTextRank SpaCy extension, TF-IDF using GenSim

- Spell Check: PyEnchant, SymSpell Python ports



## References

S. Bird, E. Klein, and E. Loper. Natural Language Processing with Python – Analyzing Text with the Natural Language Toolkit. Available online: https://www.nltk.org/book/.

[Deep Learning for Natural Language Processing (NLP)](https://machinelearningmastery.com/start-here/#nlp)

[How to Get Started with Deep Learning for Natural Language Processing](https://machinelearningmastery.com/category/natural-language-processing/)


[Natural Language Processing (NLP) in Python — Simplified](https://medium.com/@bedourabed/natural-language-processing-nlp-in-python-simplified-b96b89c8be93)


[A Detailed, Novice Introduction to Natural Language Processing (NLP)](https://towardsdatascience.com/a-detailed-novice-introduction-to-natural-language-processing-nlp-90b7be1b7e54)

[The Ultimate Guide To Different Word Embedding Techniques In NLP](https://www.kdnuggets.com/2021/11/guide-word-embedding-techniques-nlp.html)

[NLP Techniques Every Data Scientist Should Know](https://towardsdatascience.com/6-nlp-techniques-every-data-scientist-should-know-7cdea012e5c3)


[The Current Conversational AI & Chatbot Landscape](https://cobusgreyling.medium.com/the-current-conversational-ai-chatbot-landscape-c147e9bcc01b)

