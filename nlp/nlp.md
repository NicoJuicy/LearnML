# Natural Language Processing (NLP)

<!-- MarkdownTOC -->

- Overview
- NLP Cheatsheet
- Terminology
- Pre-Processing
- Applications
- Introduction to NLP
- What is NLP?
- Challenges in Natural Language Processing
- Building a Natural Language Processor
- Tokenization, Stemming, and Lemmatization
  - Tokenization
  - Stemming
  - Lemmatization
- Data Chunking
  - Example: Building Virtual Display Engines on Google Colab
- Topic Modeling and Identifying Patterns in Data
- Guide to NLP
- Bag of Words
- TF-IDF
- Tokenization
- Stop Words Removal
- Stemming
- Lemmatization
- Topic Modeling
- Text Data Pipeline
- Key Findings
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
  - HuggingFace Transformers for NLP With Python
  - Sentiment Analysis using Python
  - Sentiment Analysis using AutoML
- Applications
- NLP Pretrained Models
- NLTK
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

[NLTK cheatsheet](https://medium.com/nlplanet/two-minutes-nlp-nltk-cheatsheet-d09c57267a0b)


# Terminology

Starting with the smallest unit of data, a **character** is a single letter, number, or punctuation. 

A **word** is a list of characters and a **sentence** is a list of words. 

A **document** is a list of sentences and a **corpus** is a list of documents.

## Pre-Processing

Pre-processing is perhaps the most important step to a NLP project which involves cleaning your inputs so your models can ignore the noise and focus on what matters most. 

A strong pre-processing pipeline will improve the performance of all your models.

Below are some common pre-processing steps:

- **Segmentation:** Given a long list of characters, we might separate documents by white space, sentences by periods, and words by spaces. 

  Implementation details will vary based on the dataset.

- **Make Lowercase:** Just make everything lowercase. Capitalization generally does not add value and makes string comparison trickier. 

- **Remove Punctuation:** We may want to remove commas, quotes, and other punctuation that does not add to the meaning.
 
- **Remove Stopwords:** Stopwords are words like ‘she’, ‘the’, and ‘of’ that do not add to the meaning of a text, and can distract from the more relevant keywords.

- **Remove Other:** Depending on your application, you may want to remove certain words that do not add value. 

  For example, if evaluating course reviews, words like ‘professor’ and ‘course’ may not be useful.

- **Stemming/Lemmatization:** Both stemming and lemmatization generate the root form of inflected words (running to run). 

  Stemming is faster but does not guarantee the root is an English word. 
  
  Lemmatization uses a corpus to ensure the root is a word, at the expense of speed.

- **Part of Speech Tagging:** POS tagging marks words with their part of speech (nouns, verbs, prepositions) based on definition and context. 

  For example, we can focus on nouns for keyword extraction.

## Applications

Now that we have discussed pre-processing methods and Python libraries, we can put it all together with a few examples. 

For each, we will cover a couple of NLP algorithms, pick one based on our rapid development goals, and create a simple implementation using one of the libraries.


----------


# Introduction to NLP

## What is NLP?

**Natural Language Processing (NLP)** is a method within Artificial Intelligence (AI) that works with the analysis and building of intelligent systems that can function in languages that humans speak. 

- **Natural Language Understanding (NLU):** The understanding phase is responsible for mapping the input that is given in natural language to a beneficial representation. 

  NLU also analyzes different aspects of the input language that is given to the program.

- **Natural Language Generation (NLG):** The generation phase of the processing is used in creating Natural Languages from the first phase. 

  Generation starts with Text Planning which is the extraction of relevant content from the base of knowledge. 
  
  Next, the Sentence Planning phase chooses the words that will form the sentence. 
  
  Finally, the Text Realization phase is the final creation of the sentence structure.


## Challenges in Natural Language Processing

- **Lexical Ambiguity:** The first level of ambiguity that occurs generally in words alone. For example, when a code is given a word like "board" it would not know whether it is a noun or a verb which causes ambiguity in the processing of this piece of code.

- **Syntax Level Ambiguity:** This type of ambiguity involves the way phrases sound in comparison to how the machine perceives it. For example, a sentence such as "He raised the scuttle with a blue cap" could mean one of two things: Either he raised a scuttle with the help of a blue cap, or he raised a scuttle that had a red cap.

- **Referential Ambiguity:** This involves references made using pronouns. For instance, two girls are running on the track. Suddenly, she says, ‘I am exhausted’. It is not possible for the program to interpret, who out of the two girls is tired.


## Building a Natural Language Processor

There are a total of five execution steps when building a Natural Language Processor:

1. **Lexical Analysis:** Processing of Natural Languages by the NLP algorithm starts with identifying and analyzing the input words’ structure. This part is called Lexical Analysis and Lexicon stands for an anthology of the various words and phrases used in a language. It is dividing a large chunk of words into structural paragraphs and sentences.

2. **Syntactic Analysis/Parsing:** Once the sentences’ structure is formed, syntactic analysis works on checking the grammar of the formed sentences and phrases. It also forms a relationship among words and eliminates logically incorrect sentences. For instance, the English Language analyzer rejects the sentence, ‘An umbrella opens a man’.

3. **Semantic Analysis:** In the semantic analysis process, the input text is now checked for meaning, i.e., it draws the exact dictionary of all the words present in the sentence and subsequently checks every word and phrase for meaningfulness. This is done by understanding the task at hand and correlating it with the semantic analyzer. For example, a phrase like ‘hot ice’ is rejected.

4. **Discourse Integration:** The discourse integration step forms the story of the sentence. Every sentence should have a relationship with its preceding and succeeding sentences. These relationships are checked by Discourse Integration.

5. **Pragmatic Analysis:** Once all grammatical and syntactic checks are complete, the sentences are now checked for their relevance in the real world. During Pragmatic Analysis, every sentence is revisited and evaluated once again, this time checking them for their applicability in the real world using general knowledge.


## Tokenization, Stemming, and Lemmatization

### Tokenization

**Tokenization** or word segmentation is the process that breaks the sequence into smaller units called tokens in order to read and understand the sequence of words within the sentence, 

The tokens can be words, numerals, or even punctuation marks. 

Here is a sample example of how Tokenization works:

```
  Input: Cricket, Baseball and Hockey are primarly hand-based sports.

  Tokenized Output: “Cricket”, “Baseball”, “and”, “Hockey”, “are”, “primarily”, “hand”, “based”, “sports”
```

The start and end of sentences are called **word boundaries** which are used to understand the word boundaries of the given sentence(s).

- **Sent_tokenize package:** This package performs sentence tokenization and converts the input into sentences.

- *Word_tokenize package:* Similar to sentence tokenization, this package divides the input text into words. 

- **WordPunctTokenizer package:** In addition to the word tokenization, this package also works on punctuation marks as a token. 

```py 
    from nltk.tokenize import sent_tokenize
    from nltk.tokenize import word_tokenize
    from nltk.tokenize import WordPuncttokenizer
```

### Stemming

When studying the languages that humans use in conversations, some variations occur due to grammatical reasons.

For example, words such as virtual, virtuality, and virtualization all basically mean the same in but can have different meaning in varied sentences. 

For NLTK algorithms to work correctly, they must understand these variations. 

**Stemming** is a heuristic process that understands the word’s root form and helps in analyzing its meanings.

- **PorterStemmer package:** This package is built into Python and uses Porter’s algorithm to compute stems. Basically, the process is to take an input word of "running" and produce a stemmed word "run" as the output of the algorithm. 

- **LancasterStemmer package:** The functionality of the Lancaster stemmer is similar to Porter’s algorithm but has a lower level of strictness. It only removes the verb portion of the word from its source. 
  
  For example, the word ‘writing’ after running through the Lancaster algorithm returns ‘writ’. 

- **SnowballStemmer package:** This also works the same way as the other two and can be imported using the command . These algorithms have interchangeable use cases although they vary in strictness.

```py
    from nltk.stem.porter import PorterStemmer
    from nltk.stem.lancaster import LancasterStemmer
    from nltk.stem.snowball import SnowballStemmer
```

### Lemmatization

Adding a morphological detail to words helps in extracting their base forms which is performed using lemmatization. 

Both vocabulary and morphological analysis result in lemmatization. 

This procedure aims to remove inflectional endings. The attained base form is called a lemma.

- **WordNetLemmatizer:** The wordnet function extracts the base form of a word depending on whether the word is being used as a noun or pronoun. 

```py
    from nltk.stem import WordNetLemmatizer
```

## Data Chunking

**Chunking** is the process of dividing data into chunks which is important in NLP. 

The primary function of chunking is to classify different parts of speech and short word phrases such as noun phrases. 

After tokenization is complete and input is divided into tokens, chunking labels them for the algorithm to better understand them. 

Two methodologies are used for chunking and we will be reading about those below:

- **Chunking Up:** Going up or chunking upwards is zooming out on the problem. In the process of chunking up, the sentences become abstract and individual words and phrases of the input are generalized. 

  For example, the question "What is the purpose of a bus?"" after chunking up will answer "Transport"

- **Chunking Down:** The opposite of chunking up. During downward chunking, we move deeper into the language and objects become more specific. 

  For example, "What is a car?"" will yield specific details such as color, shape, brand, size, etc. of the car post being chunked down.

### Example: Building Virtual Display Engines on Google Colab

**Noun-Phrase Chunking:** In the code below, we will perform Noun-Phrase (NP) chunking where we search for chunks corresponding to individual noun phrases. 

To create an NP-chunker, we will define a chunk grammar rule (shown in the code below). 

The flow of the algorithm will be as follows:


## Topic Modeling and Identifying Patterns in Data

Documents and discussions are usually revolve around topics. 

The base of every conversation is one topic and discussions revolve around it. 

For NLP to understand and work on human conversations, it needs to derive the topic of discussion within the given input. 

To compute the topic, algorithms run pattern matching theories on the input to determine the topic which is called **topic modeling**. 

Topic modeling is used to uncover the hidden topics/core of documents that need processing.

Topic modeling is used in the following scenarios:

- **Text Classification:** This can improve the classification of textual data since modeling groups similar words, nouns, and actions together which does not use individual words as singular features.

- **Recommender Systems:** Systems based on recommendations rely on building a base of similar content. Therefore, topic modeling algorithms can best utilize recommender systems by computing similarity matrices from the given data.



----------


# Guide to NLP

The unigram model is also called the **bag-of-words** model.

NLP project to perform Information Extraction including Named Entity Recognition (NER) using simple regex named entity chunkers and taggers using Python and NLTK

NLP project to categorize and tag words (N-Gram Tagging) and perform feature extraction using Python

NLP project to  create an embedding from one of the texts in the Gutenberg corpus and compute some statistics related to the embedding using the Gensim library

## Bag of Words

The **bag-of-words** (BoW) is a commonly used model that allows you to count all words in a piece of text. 

BoW creates an occurrence matrix containing word frequencies or occurrences for the sentence or document (disregarding grammar and word order) which are used as features for training a classifier.

BoW has several downsides such as the absence of semantic meaning and context and the fact that stop words (like “the” or “a”) add noise to the analysis and some words are not weighted accordingly (“universe” weights less than the word “they”).

However, there are techniques to overcome these issues.


**Bag of Words:** Converting words to numbers with no semantic information. 

BoW is simply an unordered collection of words and their frequencies (counts) where the tokens (words) have to be 2 or more characters in length.


## TF-IDF

**TF-IDF:** Converting the words to numbers or vectors with some weighted information.

In Term Frequency-Inverse Document Frequency (TF-IDF), some semantic information is collected by giving importance to uncommon words than common words.

Instead of giving more weight to words that occur more frequently, TF-IDF gives a higher weight to words that occur less frequently (across the entire corpus). 

When have more domain-specific language in your text, this model performs better by giving weight to these less frequently occurring words. 


## Tokenization

**Tokenization** the process of segmenting running text into sentences and words. 

In essence, tokenization is the task of cutting a text into pieces called tokens and at the same time throwing away certain characters such as punctuation. 

Tokenization can remove punctuation too, easing the path to a proper word segmentation but also triggering possible complications. In the case of periods that follow abbreviation (e.g. dr.), the period following that abbreviation should be considered as part of the same token and not be removed.

The tokenization process can be particularly problematic when dealing with biomedical text domains which contain lots of hyphens, parentheses, and other punctuation marks.

## Stop Words Removal

Stop words removal includes getting rid of common language articles, pronouns, and prepositions such as “and”, “the” or “to” in English. 

In the process, some very common words that appear to provide little or no value to the NLP objective are filtered and excluded from the text to be processed, removing widespread and frequent terms that are not informative about the corresponding text.

Stop words can be safely ignored by carrying out a lookup in a pre-defined list of keywords, freeing up database space and improving processing time.

**There is no universal list of stop words.** 

A stop words list can be pre-selected or built from scratch. A potential approach is to begin by adopting pre-defined stop words and add words to the list later on. Nevertheless, it seems that the general trend has been to move from the use of large standard stop word lists to the use of no lists at all.

The problem is that stop words removal can remove relevant information and modify the context in a given sentence.

For example, if we are performing a sentiment analysis we might throw our algorithm off track if we remove a stop word like “not”. Under these conditions, you might select a minimal stop word list and add additional terms depending on your specific objective.


## Stemming

Stemming refere to the process of slicing the end or the beginning of words with the intention of removing _affixes_ (lexical additions to the root of the word).

The problem is that affixes can create or expand new forms of the same word called _inflectional affixes_ or even create new words themselves called _derivational affixes_. 

In English, prefixes are always derivational (the affix creates a new word as in the example of the prefix “eco” in the word “ecosystem”), but suffixes can be derivational (the affix creates a new word as in the example of the suffix “ist” in the word “guitarist”) or inflectional (the affix creates a new form of word as in the example of the suffix “er” in the word “faster”).

So if stemming has serious limitations, why do we use it? 

- Stemming can be used to correct spelling errors from the tokens. 
- Stemmers are simple to use and run very fast (they perform simple operations on a string).


## Lemmatization

The objective of **Lemmatization** is to reduce a word to its base form and group together different forms of the same word. 

For example, verbs in past tense are changed into present tense (e.g. “went” is changed to “go”) and synonyms are unified (such as “best” is changed to “good”). Thus, standardizing words with similar meaning to their root.

Although it seems closely related to the stemming process, lemmatization uses a different approach to reach the root forms of words.

Lemmatization resolves words to their dictionary form (known as lemma) which requires detailed dictionaries that the algorithm can use to link words to their corresponding lemmas.

Lemmatization also takes into consideration the context of the word to solve other problems such as disambiguation which means it can discriminate between identical words that have different meanings depending on the specific context. 

For example, think about words like “bat” (which can correspond to the animal or to the metal/wooden club used in baseball) or “bank” (corresponding to the financial institution or to the land alongside a body of water). 

By providing a part-of-speech parameter to a word (noun, verb, etc.) it is possible to define a role for that word in the sentence and remove disambiguation.

Thus, lemmatization is a much more resource-intensive task than performing a stemming process. At the same time, since it requires more knowledge about the language structure than a stemming approach, it demands more computational power than setting up or adapting a stemming algorithm.


## Topic Modeling

**Topic Modeling** (TM) is a method for discovering hidden structures in sets of texts or documents. 

In essence, TM clusters text to discover latent topics based on their contents, processing individual words and assigning them values based on their distribution. 

TM is based on the assumptions that each document consists of a mixture of topics and that each topic consists of a set of words which means that if we can spot these hidden topics we can unlock the meaning of our texts.

From the universe of topic modelling techniques, _Latent Dirichlet Allocation (LDA)_ is perhaps the most commonly used. 

LDA is a relatively new algorithm (invented less than 20 years ago) that works as an unsupervised learning method that discovers different topics underlying a collection of documents.

In unsupervised learning methods, there is no output variable to guide the learning process and data is explored by algorithms to find patterns. Thus, LDA finds groups of related words by:

Unlike other clustering algorithms such as K-means that perform hard clustering (where topics are disjointed), LDA assigns each document to a mixture of topics which means that each document can be described by one or more topics (say Document 1 is described by 70% of topic A, 20% of topic B, and 10% of topic C) and reflect more realistic results.

Topic modeling is extremely useful for classifying texts, building recommender systems (recommend based on your past readings) or even detecting trends in online publications.



----------


# Text Data Pipeline

These are some text preprocessing steps that you can add or remove as per the dataset you have:

1. Remove newlines and tabs
2. Strip HTML Tags
3. Remove Links
4. Remove Whitespaces

What are the main NLP text preprocessing steps?

The below list of text preprocessing steps is really important and I have written all these steps in a sequence how they should be.

1. Remove Accented Characters
2. Case Conversion
3. Reducing repeated characters and punctuations
4. Expand Contractions
5. Remove Special Characters
6. Remove Stopwords
7. Correcting Mis-spelled words
8. Lemmatization / Stemming


## Key Findings

The data cleaning step entirely depends on the type of Dataset. Depending on the data, more steps can be included. 

It is a must to remove extra spaces so as to reduce file size.




----------


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


### HuggingFace Transformers for NLP With Python

This article [8] explores the use of a simple pre-trained HuggingFace transformer language model for some common NLP tasks in Python.

- Text Classification
- Named Entity Recognition
- Text Summarization


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

[From raw text to model prediction in under 30 lines of Python using Atom](https://towardsdatascience.com/from-raw-text-to-model-prediction-in-under-30-lines-of-python-32133d853407)

[Powerful Twitter Sentiment Analysis in Under 35 Lines of Code](https://medium.com/thedevproject/powerful-twitter-sentiment-analysis-in-under-35-lines-of-code-a80460db24f6)

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


----------


## NLP Pretrained Models

- Polyglot
- SpaCy
- GenSim
- Pattern
- clean-text


## [NLTK](https://www.nltk.org)

[NLTK Book](https://www.nltk.org/book/)

Natural language toolkit (NLTK) is by far the most popular platform for building NLP related projects.

NLTK is also an open-source library and is available for almost every kind of operating system. 




## References

[1] S. Bird, E. Klein, and E. Loper. Natural Language Processing with Python – Analyzing Text with the Natural Language Toolkit. Available online: https://www.nltk.org/book/.

[2] [Deep Learning for Natural Language Processing (NLP)](https://machinelearningmastery.com/start-here/#nlp)

[3] [How to Get Started with Deep Learning for Natural Language Processing](https://machinelearningmastery.com/category/natural-language-processing/)

[4] [Natural Language Processing (NLP) in Python — Simplified](https://medium.com/@bedourabed/natural-language-processing-nlp-in-python-simplified-b96b89c8be93)

[5] [A Detailed, Novice Introduction to Natural Language Processing (NLP)](https://towardsdatascience.com/a-detailed-novice-introduction-to-natural-language-processing-nlp-90b7be1b7e54)

[6] [The Ultimate Guide To Different Word Embedding Techniques In NLP](https://www.kdnuggets.com/2021/11/guide-word-embedding-techniques-nlp.html)

[7] [NLP Techniques Every Data Scientist Should Know](https://towardsdatascience.com/6-nlp-techniques-every-data-scientist-should-know-7cdea012e5c3)

[8] [Exploring HuggingFace Transformers For NLP With Python](https://medium.com/geekculture/exploring-huggingface-transformers-for-nlp-with-python-5ae683289e67)


[Natural Language Processing (NLP): Don’t Reinvent the Wheel](https://towardsdatascience.com/natural-language-processing-nlp-dont-reinvent-the-wheel-8cf3204383dd)

[Guide to Natural Language Processing (NLP)](https://towardsdatascience.com/your-guide-to-natural-language-processing-nlp-48ea2511f6e1)

[Cleaning and Preprocessing Text Data by Building NLP Pipeline](https://towardsdatascience.com/cleaning-preprocessing-text-data-by-building-nlp-pipeline-853148add68a)

[Difference between Bag of Words (BOW) and TF-IDF in NLP with Python](https://pub.towardsai.net/difference-between-bag-of-words-bow-and-tf-idf-in-nlp-with-python-97d3e75a9fd)


[NLP: Classification and Recommendation Project](https://towardsdatascience.com/nlp-classification-recommendation-project-cae5623ccaae?gi=cb49766e5c29)

[Top 7 Applications of NLP (Natural Language Processing)](https://www.geeksforgeeks.org/top-7-applications-of-natural-language-processing/)

[The Current Conversational AI & Chatbot Landscape](https://cobusgreyling.medium.com/the-current-conversational-ai-chatbot-landscape-c147e9bcc01b)

