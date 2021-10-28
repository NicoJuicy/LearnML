# Natural Language Processing (NLP)

## Overview

Natural Language Processing (NLP) is a category of ML concerned with the analysis and building of intelligent systems that can function in languages that humans speak. 

Processing of language is needed when a system wants to work based on input from a user in the form of text or speech and the user is adding input in regular use English.

- Natural Language Understanding (NLU): The understanding phase of the processing is responsible for mapping the input that is given in natural language to a beneficial representation. It also analyzes different aspects of the input language that is given to the program.

- Natural Language Generation (NLG): The generation phase of the processing is used in creating Natural Languages from the first phase. Generation starts with Text Planning, which is the extraction of relevant content from the base of knowledge. Next is the Sentence Planning phase where required words that will form the sentence are chosen. Ending with Text Realization is the final creation of the sentence structure.

## Challenges in Natural Language Processing

- Lexical Ambiguity: This is the first level of ambiguity that occurs generally in words alone. For instance, when a code is given a word like ‘board’ it would not know whether to take it as a noun or a verb. This causes ambiguity in the processing of this piece of code.

- Syntax Level Ambiguity: This is another type of ambiguity that has more to do with the way phrases sound in comparison to how the machine perceives it. For instance, a sentence like, ‘He raised the scuttle with a blue cap’. This could mean one of two things. Either he raised a scuttle with the help of a blue cap, or he raised a scuttle that had a red cap.

- Referential Ambiguity: References made using pronouns constitute referential ambiguity. For instance, two girls are running on the track. Suddenly, she says, ‘I am exhausted’. It is not possible for the program to interpret, who out of the two girls is tired.




## Common Tasks

Some NLP use cases and tasks are:

- Sentiment Analysis: to understand the sentiment (negative, positive, neutral) a certain document/text holds.

Example: social media posts about Climate Change.

- Topic Modeling: to draw clusters or organize the data based on the topics (goal is to learn these topics) it contains.

Example: an insurance company wants to identify fraudulent claims by classifying causes into main labels and then further analyze the ones containing suspicious content/topics.

- Text Generation: to create new data (textual mainly) based on previous examples from the same domain. Example: chatbots, quotes, email replies, etc.

Example: chatbots, quotes , email replies, …, etc.

- Machine Translation → to automatically convert from one language to another.

Example: English to German. 


## NLP Workflow 

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

The performance of NER depends heavily on the training data used to develop the model. The more relevant the training data to the actual data, the more accurate the results will be.

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


## Applications

- Document Clustering

The general idea with document clustering is to assign each document a vector representing the topics discussed. 

- Sentiment Analysis

- Keyword Extraction: Named Entity Recognition (NER) using SpaCy, Rapid Automatic Keyword Extraction (RAKE) using ntlk-rake

- Text Summarization: TextRank (similar to PageRank) using PyTextRank SpaCy extension, TF-IDF using GenSim

- Spell Check: PyEnchant, SymSpell Python ports


## References

[Natural Language Processing (NLP) in Python — Simplified](https://medium.com/@bedourabed/natural-language-processing-nlp-in-python-simplified-b96b89c8be93)

[Cleaning & Preprocessing Text Data by Building NLP Pipeline](https://towardsdatascience.com/cleaning-preprocessing-text-data-by-building-nlp-pipeline-853148add68a)

[A Detailed, Novice Introduction to Natural Language Processing (NLP)](https://towardsdatascience.com/a-detailed-novice-introduction-to-natural-language-processing-nlp-90b7be1b7e54)

[NLP Techniques Every Data Scientist Should Know](https://towardsdatascience.com/6-nlp-techniques-every-data-scientist-should-know-7cdea012e5c3)


