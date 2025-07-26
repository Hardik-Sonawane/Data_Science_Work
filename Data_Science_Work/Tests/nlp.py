#####################   Test 10 ###############################3
###############################################################
""" 
Que1.
 Write a NLTK program to omit some given stop words 
from the stopwords list. 
Stopwords to omit : 'again', 'once', 'from'
"""
#Solution :
import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
omit_swords = {'again','once','from'}
New_stop_words = stop_words - omit_swords
print(f"Updated stopwords list: {sorted(New_stop_words)}")
##################################################
##################################################
"""
Que.4
 Implement basic text preprocessing steps on a dataset, 
including tokenization, lowercasing, removing stopwords, 
punctuation, and special characters. 
text = "Hello! This is a sample text. Let's tokenize it, remove stopwords and 
punctuations. Hope you all are doing well!" 
"""
# Solution
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
text = "Hello! This is a sample text. Let's tokenize it, remove stopwords and punctuations. Hope you all are doing well!"
text = text.lower()
tokens = word_tokenize(text)
stop_words = set(stopwords.words('english'))
tokens = [word for word in tokens if word not in stop_words]
tokens = [word for word in tokens if word.isalnum()]
print(tokens)

#############################################
##############################################
"""
Que.3
 Create a text summarization model using BART to generate 
summaries from news articles. 
Note : Use cnn_dailymail dataset 
"""
#Solution

from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset

model_name = 'facebook/bart-large-cnn'
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)
dataset = load_dataset('cnn_dailymail', '3.0.0', split='test')

def summarize_article(article_text, max_len=130, min_len=30):
    
    inputs=tokenizer([article_text], max_length=1024, return_tensors='pt', truncation=True)
    
    summary_ids=model.generate(inputs['input_ids'], num_beams=4, max_length=max_len, min_length=min_len, length_penalty=2.0, early_stopping=True)
    summary=tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
article_text = dataset[0]['article']
summary = summarize_article(article_text)

print("Originalwala Article:\n", article_text)
print("\n Summary:\n", summary)
##################################################
"""
Que.2
 Build a Named Entity Recognition (NER) model that identifies 
entities (people, locations, organizations) in text using 
advanced models like LSTM-CRF or BERT. Compare the 
performance of these models. 
Note : Use the ‘conll2003’ dataset from dataset library for this problem
"""
#Solution 

##################################################
"""
Que.5.
Perform basic sentiment analysis on text using a Bag-of
Words model. Build a classifier to predict whether a review is 
positive or negative. 
Dataset : books.csv
"""

import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.accuracy_score import accuracy_score

df=pd.read_csv("C:/9-pca_svd/books.csv")

print("preview of the datasset :")
print(df.head())


stop_words = set(stopwords.words('english'))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\d+', '', text)  
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

df['cleaned_review'] = df['review'].apply(preprocess_text)

X = df['cleaned_review']  
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(max_features=5000)  
X_train_bow = vectorizer.fit_transform(X_train)
X_test_bow = vectorizer.transform(X_test)


classifier = LogisticRegression()
classifier.fit(X_train_bow, y_train)


y_pred = classifier.predict(X_test_bow)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')


print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))





