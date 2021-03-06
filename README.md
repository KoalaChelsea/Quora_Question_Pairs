# Identify Quora Question Pairs With The Same Intents 
___Dance Squad: Jingjing Lin, Jiaqi Tang, Yingjie(Chelsea) Wang, Xinyi Ye___

### Quick menu

* [Roles and Responsibilities](#Roles-and-Responsibilities)
* [Motivation](#Motivation)
* [Relevant Background Research](#Relevant-Background-Research)
* [Data Collection](#Data-Collection)
* [Methods](#Methods)
* [Procedure](#Procedure)
* [Results](#Results)
* [Analysis](#Analysis)
* [References](#References)


## Roles and Responsibilities
___Student group members and which student is responsible for what parts?___

Jiaqi Tang:
- Brought up project ideas and data and helped figure out the general process of methods. 
- Performed two-component LSA after TF-IDF for feature engineering.
- Conducted similarity matrix, which including cosine distance Manhattan distance and Euclidean distance for LSA components of quora question pairs.
- Calculated the log loss after applying the similarity matrix.
- Used machine learning methods: Random forest and SVR for reducing the log loss from the similarity matrix for optimization.

Jingjing Lin: 
- Performed the TF-IDF transformation from a provided matrix of counts and measured cosine similarity between quora question pairs. 
- Applied Logistic Regression to reduce log loss. 
- Helped write the README.

Yingjie(Chelsea) Wang: 
- Applied multiple classifiers on TFIDF + Cosine Similarity, such as MultinomialNB and RandomForestClassifier and Decision Tree. 
- Visualized the evaluation results and make the comparison on log loss. 
- Helped organize and write the README.   

Xinyi Ye:
- Conducted EDA and created histograms, word cloud, etc.
- Processed data (tokenization, stemming) and performed Bag-of-Words transformation.
- Defined similarity calculation functions for five distances, which are Cosine, Manhattan, Euclidean, Jaccard, and Minkowski.
- Conducted baseline assessments with these five similarity functions and calculated log loss of each. 
- Performed SVR, Random Forest Regressor, and Decision Tree Regressor with the similarity matrix and output log loss for further comparison.

## Motivation
___What is the research question and why is it worth asking?___

### Research Question
Comparing different similarity calculations with embedding methods to improve the accuracy of identifying Quora question pairs.

### Motivation for Project
Text Similarity plays a significant role in both text-related research and real-world applications. Measuring the similarity between terms, sentences, paragraphs, and documents has become a powerful tool to conduct future machine learning research. Also, websites like Quora, which is highly sensitive to the textual data, using NLP methods to identify the duplicated information would improve the user experience and "fold" the memory storage for the company.      

## Relevant Background Research
___What prior work does your proposal rely upon or sets the context for your question?___
#### Similarity methods with supervised learning:
To calculate the similarity of pairs, there are useful many measurement functions such as Cosine Distance, Euclidean Distance, Manhattan Distance, Jaccard Distance, Minkowski Distance, etc. In previous studies, they used supervised training combined with similarity methods to help sentence embeddings learn the meaning of a sentence more directly. For example, Smooth Inverse Frequency + Cosine Similarity.

#### Word Embedding methods
Word embeddings tend to popular in Natural Language Processing, especially for measuring the similarity of the semantic meaning of words[Adrien Sieg,2018]. In previous studies, to measure the semantics of words and short sentences. The three known methods are LSA, Word2Vec, and GloVe have been utilized. 

#### Quora Question Pairs Identification
There are various methods and techniques have been used for identifying quora question Paris from Kaggle competition held by Quora. In this case, from this competition, most of the teams used high-level Machine learning and even deep learning methods to predict and identify quora question pairs. 

However, there are few works from Kaggle to build the methods to explain how to improve the accuracy based on NLP techniques, for example, Word embedding methods. 

Therefore, according to previous work of NLP techniques and this specific question that is quora question Paris identification. We intend to use different word embedding methods with pre-trained and to combine with similarity calculations in order to find an optimal way to measure the similarity of the semantic meaning of sentences.

## Data Collection
We acquired the dataset (https://www.kaggle.com/c/quora-question-pairs/data) from one of the Kaggle’s NLP competitions. There are two datasets in this project, including the training data and testing data. There are 402900 question pairs of training data. 36.92% of them are duplicate pairs, and 111780 questions appeared multiple times.

## Methods
### Procedure
Similar to the other Natural Language Process, there are also few steps to conduct text-similarity comparison in this project.

- Exploratory Data Analysis (EDA)
- Data Pre-processing
- Feature Engineering (using different word embedding methods and "distance" measurement)
- Model Selection and Building
- Model Comparison by Log Loss and Accuracy

#### Exploratory Data Analysis:
Before conducting any data mining procedures, we performed EDA to get a whole picture of the dataset, including identifying the duplicate pairs, checking the number of characters and words of each question as well as using WordCloud to visualize the most frequent words.

#### Pre-processing:
After that, we preprocessed the dataset using the regular expression to clean, uniform all content. We split data into training and testing datasets by random.

#### Embeddings:
In this project, three embeddings methods tend to be used to represent the word vectors: TD-IDF, TD-IDF+ISA, a bag of words.

#### Similarity and Supervised Learning Methods:
With word embeddings processing, we used distance matrix as the method for identifying the Paris of quora question (if these two questions are the same questions or not), and machine learning methods are conducted for reducing the log loss from distance matrix in order to perform optimization.

We used Cosine Distance, Manhattan distance, Jaccard Distance, and euclidean distance individually and compared with supervised learning methods with similarity measure methods: SVM/Logistic Regression + Cosine Similarity, Similarity Matrix + Random Forest/Support Vector Regression/Decision Tree Regressor, LSA + Similarity Matrix + Random Forest/Support Vector Regression.

Overall, there are seven models after combining different word embedding methods, distance matrix with different machine learning modelings.

#### Performance Evaluation:
Log loss was implemented to evaluate the performance of our results from these seven models.

## Results 
___How do you measure performance?___

After data pre-processing, we used the distance matrix as the baseline for identifying the Paris of quora question:
- Cosine: 0.961
- Manhattan: 2.208
- Euclidean: 1.830
- Jaccard: 2.821
- Minkowski：1.830

After applying different machine learning methods, the ranking of the model performance in term of log loss from best to worst is:
- Bag-of-Words + Cosine, Manhattan, Eucledian, Jaccard, Minkowst + Random Forest Regressor (0.5934)
- TF-IDF + Cosine + Logistic Regression (0.6024)
- TF-IDF + Cosine + MultinomialNB (0.6582)
- Bag-of-Words + Cosine, Manhattan, Eucledian, Jaccard, Minkowst + Regression Tree (0.7052)
- TF-IDF + Cosine, Manhattan, Eucledian + MultinomialNB + LSA + Random Forest Regressor(0.7656)
- Bag-of-Words + Cosine, Manhattan, Eucledian, Jaccard, Minkowst + SVR (0.7712)
- TF-IDF + Cosine + Random Forest (0.8032)

![picture alt](https://github.com/KoalaChelsea/Quora_Question_Pairs/blob/master/Comparison.PNG "Performance Comparison")


## Analysis

From the baseline analysis, cosine is the best similarity method for identifying Quora Paris with the lowest log loss among the five distances. However, we believe the log loss can be reduced by model blending. 
 
 We built Bag-of-Words (BoW) and TFIDF with different distance measurements (Cosine, Manhattan, Euclidean, Jaccard, and Minkowski) and tested the feature matrix in different models (Logistic regression, MultinomialNB, Random forest, and SVR) by Log Loss and Accuracy. After combining different word embedding methods, distance matrix with various machine learning modelings, the best performance comes from Bag-of-Words + Similarity Matrix + Random Forest Regressor, which has a log loss of 0.5934. Therefore, using machine learning algorithms tend to reduce log loss significantly. 

Also, from the plot above, we learn that using a similarity matrix containing different distance methods will obtain better performance, compared to the models that utilized only cosine similarity. In terms of the three embedding methods, a bag of words provides the best solution with the lowest log loss, compared to TF-IDF and LSA.

There might be more complex modeling blending that can improve the performance further. BoW and TF-IDF are two of the most common methods people use in information retrieval. Generally speaking, SVMs and Naive Bayes are more common for the classification problem; however, because their accuracy is dependent on the training data, Xgboost provided the best accuracy in this particular data set. I will suggest Xgboost as a possible further approach to improve performance.


![picture alt](https://github.com/KoalaChelsea/Quora_Question_Pairs/blob/master/poster.jpg "Poster")

## References
[1] Similarity Techniques + NLP: https://www.kaggle.com/tj2552/similarity-techniques-nlp

[2] Shashi Shankar, Aniket Shenoy(2018): Identifying Quora question pairs having the same intent

[3] Wael H. Gomaa(2013): A Survey of Text Similarity Approaches

[4] Text Similarities: Estimate the degree of similarity between two texts: https://medium.com/@adriensieg/text-similarities-da019229c894

[5] Pre-trained Word Embeddings or Embedding Layer? — A Dilemma: 
https://towardsdatascience.com/pre-trained-word-embeddings-or-embedding-layer-a-dilemma-8406959fd76c		





