# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 08:16:31 2024

@author: sonaw
"""
########################################################################################
"""
Que.4. Using the mlxtend library, write a Python program to generate association 
rules from a dataset of transactions. The program should allow setting a 
minimum support threshold and minimum confidence threshold for rule 
generation. 
transactions = [['Tea', 'Bun'], ['Tea', 'Bread'], ['Bread', 'Bun'], ['Tea', 'Bun', 
'Bread']] 
"""
#########3 Solution

# Step 1: Import required libraries
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd

######    Step2: Define the transaction data
data = [['Tea', 'Bun'], 
        ['Tea', 'Bread'], 
        ['Bread', 'Bun'], 
        ['Tea', 'Bun', 'Bread']]

###Step3:Encode the transaction data
encoder = TransactionEncoder()
data_array = encoder.fit(data).transform(data)

#######Step4: Convert the encoded data into a pandas DataFrame
df = pd.DataFrame(data_array, columns=encoder.columns_)

#####Step5: Define a function to generate association rules
def get_rules(min_support, min_confidence):
    itemsets = apriori(df, min_support=min_support, use_colnames=True)
    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    return rules

####Step 6: Set minimum support and confidence thresholds
support = 0.5
confidence = 0.7

####Step 7: Generate the association rules
rules = get_rules(support, confidence)

####Step 8: Print the generated rules
print(rules)
#########################################################################################

"""
Que.2.Output  Will be 
"""

"""
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd

transactions = [['Apple', 'Banana'], 
                ['Apple', 'Orange'], 
                ['Banana', 'Orange'], 
                ['Apple', 'Banana', 'Orange']]

te = TransactionEncoder()
transformed_data = te.fit(transactions).transform(transactions)
df = pd.DataFrame(transformed_data, columns=te.columns_)

frequent_items = fpgrowth(df, min_support=0.5, use_colnames=True)

print(frequent_items)
"""
##### Solution

#       support          itemsets
#      0     0.75          (Banana)
#      1     0.75           (Apple)
#      2     0.75          (Orange)
#      3     0.50  (Orange, Banana)
#      4     0.50   (Apple, Banana)
#      5     0.50   (Apple, Orange)

##############################################################################################################

"""
Que.5. Build a popularity-based recommendation system. The system should 
recommend movies based on their overall popularity (e.g., number of ratings 
or average rating). 
Steps: 
 Preprocess the Data: Calculate the total number of ratings and average rating for each 
movie. 
 Rank the Movies: Rank movies based on the chosen popularity metric. 
 Recommend Movies: Recommend the top N most popular movies to any user.
"""
## Solution
import pandas as pd
data = {'id': [1, 2, 3, 4, 5],
        'movie': ['Bahubali 2', 'KGF 2', 'RRR', 'Vikram', 'Maharaja'],
        'rating': [9, 8, 8.5, 9, 9.5]}
df = pd.DataFrame(data)
stats = df.groupby('movie').agg(count=('rating', 'count'), avg=('rating', 'mean')).reset_index()
stats = stats.sort_values(by='avg', ascending=False)
stats['avg'] = stats['avg'].apply(lambda x: f"{x}/10")
print(stats)

########################################################################################

"""Que.3. Build an item-based collaborative filtering recommendation engine. 
Instead of recommending items based on similar users, recommend items 
that are similar to those that a user has already interacted with. 
Steps: 
 Preprocess the Data: Create a user-item matrix where rows are users and columns are 
items (movies). 
 Compute Item Similarity: Calculate similarity between items based on user 
interactions. 
 Recommend Items: For a given user, recommend items that are similar to those the 
user has already rated highly."""

######Solution 

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load
dataset_path = "C://11-recommendation_system//imdb_top_1000.csv"
data = pd.read_csv(dataset_path)

# Preprocess
movies_df = data[['Series_Title', 'IMDB_Rating']]
movie_matrix = movies_df.pivot_table(index='Series_Title', values='IMDB_Rating').fillna(0)

# Similarity
item_similarity = cosine_similarity(movie_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=movie_matrix.index, columns=movie_matrix.index)

# Recommend
def recommend_similar_movies(movie_title, movie_matrix, item_similarity_df, top_n=5):
    if movie_title not in movie_matrix.index:
        return "Movie not found in the dataset"
    similar_movies = item_similarity_df[movie_title].sort_values(ascending=False).drop(movie_title).head(top_n)
    return similar_movies

# Example
movie_title = 'The Godfather'  # Change movie title to test others
recommended_movies = recommend_similar_movies(movie_title, movie_matrix, item_similarity_df, top_n=5)

# Output
print(f"Movies similar to '{movie_title}':")
print(recommended_movies)

############################################################################################t
"""
1. You are given a dataset of movies with various attributes like genres, 
keywords, and descriptions. Your task is to build a content-based 
recommendation engine that recommends movies similar to a given movie 
based on these attributes. 
Steps: 
 Preprocess the Data: Extract relevant features (e.g., genres, overview). 
 Vectorize the Text Data: Use TF-IDF on the overview field. 
 Compute Similarity: Use cosine similarity to find similar movies. 
 Recommend: Given a movie, recommend the top 10 most similar movies based on 
content. 
Note: Use IMDB dataset
"""
##Solution..

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
data_path = "C:\\11-recommendation_system\\IMDb_Movie_Reviews.csv"
data = pd.read_csv(data_path)

# Preprocess: Extract 'Review_Text' column and drop NaN
reviews = data[['Review_Text']].dropna()

# Vectorize: Apply TF-IDF to 'Review_Text'
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(reviews['Review_Text'])

# Similarity: Compute cosine similarity
similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Recommend: Function to get top 10 similar reviews
def recommend_reviews(input_review, reviews, similarity_matrix, top_n=10):
    # Find index of the input review
    try:
        idx = reviews[reviews['Review_Text'] == input_review].index[0]
    except IndexError:
        print("No exact match found. Here are some reviews:")
        print(reviews['Review_Text'].sample(n=5, random_state=1))
        return []
    
    # Get similarity scores for the input review
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort by similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    
    # Get review indices
    review_indices = [i[0] for i in sim_scores]
    
    # Return top-n similar reviews
    return reviews['Review_Text'].iloc[review_indices]

# Example usage
input_review = "This is an example review text."  # Replace with a review from your dataset
recommended = recommend_reviews(input_review, reviews, similarity_matrix, top_n=10)

# Output
if recommended:
    print("Similar reviews:")
    print(recommended)
###########################################################################################