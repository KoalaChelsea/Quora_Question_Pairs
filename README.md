# Identify Quora Question Pairs With The Same Intents 
## Dance Squad: Jingjing Lin, Jiaqi Tang, Yingjie(Chelsea) Wang, Xinyi Ye


### Student group members and which student is responsible for what parts?
Xinyi Ye

Jiaqi Tang

Jingjing Lin: 
- Performed the TF-IDF transformation from a provided matrix of counts and measured cosine similarity between quora question pairs. 
- Applied Logistic Regression to reduce the log loss. 
- Helped write the README.


Yingjie(Chelsea) Wang: 
- Applied multiple classifiers on TFIDF + Cosine Similarity, such as MultinomialNB and RandomForestClassifier and Decision Tree. 
- Visualized the evaluation results and make comparison on log loss. 
- Helped organize and write the README.   

### Motivation. What is the research question and why is it worth asking?
#### Research Question
Comparing different similarity calculations with embedding methods to improve the accuracy of identifying Quora question pairs.

#### Motivation 
Text Similarity plays a significant role in both text-related research and real-world applications. Measuring the similarity between terms, sentences, paragraphs, and documents has become a powerful tool to conduct future machine learning research. In addition, websites like Quora which is highly sensitive to the textual data, using NLP methods to identify the duplicated information would improve the user experience and “fold” the memory storage for the company. 		
		
#### Relevant background research. What prior work does your proposal rely upon or sets the context for your question?
##### Similarity methods with supervised learning:
To calculate the similarity of pairs, there are useful many measurement functions such as Cosine Distance, Euclidean Distance, Manhattan Distance, Jaccard Distance, Minkowski Distance, etc. In previous studies, they used supervised training combined with similarity methods to help sentence embeddings learn the meaning of a sentence more directly. For example, Smooth Inverse Frequency + Cosine Similarity.
##### Word Embedding methods
Word embeddings tend to popular in Natural Language Processing, especially for measuring the similarity of the semantic meaning of words[Adrien Sieg,2018].In previous studies, to measure the semantics of words and short sentences. The three known methods are LSA, Word2Vec and GloVe have been utilized. 

##### Quora Question Pairs Identification
There are various methods and techniques have been used for identifying quora question Paris from Kaggle competition held by Quora. In this case, from this competition, most of the teams used high-level Machine learning and even deep learning methods to predict and identify quora question pairs. 
However, there are few works from Kaggle to build the methods to explain how to improve the accuracy based on NLP techniques, for example, Word embedding methods. 
Therefore, according to previous work of NLP techniques and this specific question that is quora question Paris identification. We intend to use different word embedding methods with pre-trained and to combine with similarity calculations in order to find an optimal way to measure the similarity of the semantic meaning of sentences.

### Data (collection) 
We acquired the dataset (https://www.kaggle.com/c/quora-question-pairs/data) from one of the Kaggle’s NLP competitions. There are two datasets in this project, including the training data and testing data. There are 402900 question pairs of training data. 36.92% of them are duplicate pairs, and 111780 questions appeared multiple times.

### Methods
#### Procedure
Similar to the other Natural Language Process, there are also few steps to conduct text-similarity comparison in this project.

- Exploratory Data Analysis (EDA)
- Preprocess data
- Feature Engineering (using different word embedding methods and “distance” measurement)
- Model selection and building
- Model comparison by Log Loss and Accuracy

Before conducting any data mining procedures, we performed EDA to get a whole picture of the dataset, including identifying the duplicate pairs, checking the number of characters and words of each question as well as using Wordcloud to visualize the most frequent words. After that, we preprocessed the dataset using the regular expression to clean, uniform all content. We split data into training and testing datasets by random. We built Bag-of-Words (BoW) and TFIDF with different distance measurements (Cosine, Manhattan, Euclidean, Jaccard and Minkowski) and tested the feature matrix in different models (Logistic regression, MultinomialNB, Random forest and SVR) by Log Loss and Accuracy.

### Results (including how you are measuring performance)
### Analysis

### References
[1] Similarity Techniques + NLP: https://www.kaggle.com/tj2552/similarity-techniques-nlp
[2] Shashi Shankar, Aniket Shenoy(2018): Identifying Quora question pairs having the same intent
[3] Wael H. Gomaa(2013): A Survey of Text Similarity Approaches
[4] Text Similarities: Estimate the degree of similarity between two texts: https://medium.com/@adriensieg/text-similarities-da019229c894
[5] Pre-trained Word Embeddings or Embedding Layer? — A Dilemma: 
https://towardsdatascience.com/pre-trained-word-embeddings-or-embedding-layer-a-dilemma-8406959fd76c		





