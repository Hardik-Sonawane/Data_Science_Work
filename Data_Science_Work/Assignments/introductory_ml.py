"""
¬¬Topic: Text Mining (NLP)

1.	Business Problem
1.1.	What is the business objective?
1.2.	Are there any constraints?
------>
Objectives:::--->
The goal is to use text mining and NLP (Natural Language Processing) to turn unstructured text data into useful information. This could help businesses in several ways
Understand Customer Feelings: Analyze reviews and feedback to find out if customers are happy or unhappy.
Improve Products or Services: Use feedback to make better products or services by addressing common complaints.
Research Market Trends: Find out what people are saying about products and competitors to understand market trends.
Monitor Brand Reputation: Track mentions of the brand to manage its reputation and respond to any issues.
Automate Customer Support: Use automated systems to handle customer inquiries more efficiently.
constraints:::---->
Here are some challenges businesses might face:
Data Quality: The results depend on the quality of the data. If the data is messy or incomplete, the insights might not be accurate.
Privacy Concerns: Businesses must follow data protection laws to ensure customer data is handled properly.
Language Issues: Text can vary in slang, jargon, or languages. Models need to understand these variations to provide accurate insights.
Resource Requirements: Analyzing text data can require a lot of computer power and memory, especially with large datasets.
Complexity vs. Efficiency: More advanced models might give better results but can be harder to manage and slower to run.
Real-Time Needs: For some uses, like social media monitoring, the analysis needs to be done quickly.
Industry-Specific Knowledge: Understanding the specific industry or field is important for accurate analysis. Generic tools might not always work well.
By addressing these challenges, businesses can effectively use text mining and NLP to meet their goals.
"""
#########################################################
"""
2..
| **Name of the Feature** | **Description**                                                   | **Type**              | **Relevance**                                                   |
|-------------------------|------------------------------------------------                     |-----------------------|-----------------------------------------------------------------|
| `Text`                  | Contains the actual content of the review or comment.             | String (Text)         | Highly relevant; central to text mining tasks like sentiment analysis and topic modeling. |
| `Date`                  | Date when the review or comment was posted.                   | Date/Time             | Medium relevance; useful for analyzing trends over time but not directly used in text analysis. |
| `Sentiment`             | Sentiment label of the text (e.g., Positive, Negative, Neutral).  | Categorical           | Highly relevant; directly used for sentiment classification and evaluation of model performance. |
| `Rating`                | Numerical rating given by the customer (e.g., 1-5 stars).      | Numeric               | High relevance; provides context for sentiment and helps in cross-validating sentiment analysis. |
| `Product_ID`            | Unique identifier for the product being reviewed.              | Categorical/Numeric   | Low relevance; while useful for product-specific insights, it does not contribute directly to text analysis. |
| `User_ID`               | Unique identifier for the user who posted the review.            | Categorical/Numeric   | Low relevance; identifies the user but is not critical for text analysis. Useful for user-specific behavior analysis but not central to content analysis. |
| `Location`              | Geographic location of the user.                             | Categorical           | Medium relevance; useful for analyzing regional trends or sentiment differences, but not central to the core text analysis. |



Text: This is the primary feature for text mining. It contains the actual textual data from which insights are extracted. It is essential for tasks such as sentiment analysis, topic modeling, and keyword extraction.
Date: Useful for tracking changes and trends over time. While it doesn’t directly affect text analysis, it can provide valuable context for understanding how sentiment or feedback evolves.
Sentiment: Represents the emotional tone of the text and is crucial for sentiment analysis. It directly impacts how the model categorizes and interprets the text data.
Rating: Provides additional context that complements the sentiment. High ratings generally correlate with positive sentiment and vice versa, making it valuable for validating sentiment analysis results.
Product_ID: Identifies the product associated with the review. It is not directly involved in text mining but is useful for segmenting analysis by product and understanding product-specific feedback.
User_ID: Identifies the reviewer but does not contribute directly to the content analysis. It might be useful for tracking individual user patterns but is less relevant for text mining models focused on sentiment and content.
Location: Provides geographic context, which can help in analyzing regional sentiment trends. It adds value to the analysis but is not crucial for the core text mining tasks.

"""
#################################################
"""
3.	Data Pre-processing
3.1 Data Cleaning, Feature Engineering, etc.
3.2 Outlier Treatment
----->
3.1 Data Cleaning and Feature Engineering
Data Cleaning::::>

    Remove Unnecessary Stuff:
Special Characters: Get rid of symbols like @, #, or & that don't help with understanding the text.
HTML Tags: If your text has HTML code (like <b> or <i>), remove it.

Standardize Text:
Lowercasing: Change all text to lowercase so "Hello" and "hello" are treated the same.
Remove Punctuation: Get rid of periods, commas, and other punctuation that don't add value.

Break Down the Text:
Tokenization: Split the text into individual words or tokens (e.g., "This is great" becomes ["This", "is", "great"]).

Remove Common Words:
Stop Words: Remove common words like "and", "the", or "is" that don't carry much meaning.

Simplify Words:
Stemming: Reduce words to their base form (e.g., "running" becomes "run").
Lemmatization: Change words to their base or dictionary form (e.g., "better" becomes "good").
Fix Spelling Mistakes:
Correct any spelling errors in your text.



Feature Engineering:::>

 Count Words:
Term Frequency (TF): Count how often each word appears in a text to see which words are important.

Weight Words:
TF-IDF: Adjust word counts by how common or rare they are across all texts. This helps highlight important words and reduce the impact of common ones.

Word Combinations:
N-grams: Create groups of words (like pairs or triples) to capture word sequences (e.g., "not bad" becomes a bigram).

Convert Words to Numbers:
Word Embeddings: Turn words into numerical vectors that represent their meanings (using methods like Word2Vec).

Identify Parts of Speech:
POS Tagging: Label words with their roles (e.g., noun, verb) to understand grammar.
Find Named Entities:

NER: Identify and label names of people, places, or organizations.


Outlier Treatment::::>
Handling Outliers:

Spot Unusual Data:
Rare Words or Phrases: Look for words that appear very rarely and might be errors.
Document Length: Check if some documents are much shorter or longer than usual.

Deal with Outliers:
Remove: Delete irrelevant or odd documents or words.
Adjust: Modify or correct data that doesn’t fit well with the rest.
Check Context:
See if outliers make sense in their context or if they should be adjusted or removed.

Simple Steps:
Load Data: Start by getting your text data ready.
Clean Data: Remove unwanted characters, standardize text, and fix errors.
Engineer Features: Create useful features from the text, like word counts or combinations.
Identify Outliers: Look for unusual or irrelevant data.
Handle Outliers: Decide whether to remove or adjust these unusual pieces of data.
"""
##########################################################
"""
4	Exploratory Data Analysis (EDA):
4.1.	Summary
4.2.	Univariate analysis
4.3.	Bivariate analysis
---->

EDA helps you understand your data before doing any complex analysis. Here’s a simple breakdown:
4.1 Summary
Overview: Look at the size of your data (how many rows and columns).
Types of Data: Check what kind of data each column holds (e.g., text, numbers).
Missing Data: Find any missing values.
Basic Stats: For numbers, look at average, minimum, maximum, and spread. For text, count how many times each word appears.
4.2 Univariate Analysis
Text Data:

Word Frequency: Find out which words appear the most.
Text Length: Check how long the text is on average.
Numerical Data:

Histograms: See how numerical data is spread out.
Box Plots: Find out if there are any unusual data points (outliers).
Categorical Data:

Bar Charts: Show how often each category occurs.
Pie Charts: Display the proportion of each category.
4.3 Bivariate Analysis
Text Data:

Word Pairs: See if certain words often appear together.
Length vs. Sentiment: Check if the length of text affects its sentiment.
Numerical Data:

Scatter Plots: Show the relationship between two numbers.
Correlation: Measure how strongly two numerical variables are related.
Categorical Data:

Cross-Tabs: Show how categories from two variables interact.
Stacked Bar Charts: Compare categories across different groups.
In Summary:

Summary: Get a quick overview and basic stats of your data.
Univariate: Look at individual features.
Bivariate: Explore how features relate to each other.

"""
###############################################################
"""

5.	Model Building
5.1	Extract text data from websites such as Amazon, Snapdeal, IMDB, Twitter, etc.
5.2	Clean the data and build a word cloud for both positive and negative words. Perform Sentiment Analysis as well. 
5.3	Briefly explain the model output in the documentation.
"""
#---->
import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.data import find
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

#requests: For making HTTP requests to fetch web pages.
#BeautifulSoup: For parsing HTML and extracting data.
#re: For regular expression operations to clean text.
#nltk: For natural language processing tasks like tokenization, stopword removal, and stemming.
#wordcloud: For generating word clouds.
#matplotlib: For plotting visualizations.
#TextBlob: For performing sentiment analysis.

# Download necessary NLTK data
try:
    find('corpora/stopwords.zip')
except LookupError:
    from nltk import download
    download('stopwords')
#Checks if the stopwords data is available, and if not, downloads it.
def fetch_reviews(product_url, num_reviews=10):
    reviews = []
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    while len(reviews) < num_reviews:
        try:
            response = requests.get(product_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract review text
            review_elements = soup.find_all('span', {'data-asin-review-id': True})
            for element in review_elements:
                if len(reviews) >= num_reviews:
                    break
                review_text = element.get_text()
                reviews.append(review_text)

            # Check if there's a next page
            next_page = soup.find('li', {'class': 'a-last'})
            if next_page and next_page.a:
                next_page_url = next_page.a['href']
                product_url = f"https://www.amazon.in{next_page_url}"
            else:
                break
        except requests.RequestException as e:
            print(f"Request error: {e}")
            break
    
    return reviews
#Fetch Reviews: Scrapes the reviews from the specified Amazon product page. It handles pagination by navigating to the next page if available.
def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(tokens)
#Clean Text: Cleans the review text by removing HTML tags, URLs, punctuation, converting to lowercase, removing stopwords, and applying stemming.
def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns a value between -1 (negative) and 1 (positive)
#Analyze Sentiment: Uses TextBlob to analyze sentiment, returning a polarity score which indicates the sentiment of the text.
# Example usage
product_url = 'https://www.amazon.in/JBL-C100SI-Ear-Headphones-Black/dp/B01DEWVZ2C/?_encoding=UTF8&pd_rd_w=I1AmU&content-id=amzn1.sym.a490a4f5-1c2a-464e-8df8-c9593e3e7b44%3Aamzn1.symc.4d65739a-ff51-46fc-ad8f-587c12f6fb5e&pf_rd_p=a490a4f5-1c2a-464e-8df8-c9593e3e7b44&pf_rd_r=RAGE5E2YNFWFCMW1AXQG&pd_rd_wg=d4zrI&pd_rd_r=4b0c2b5f-d4e8-4f78-8f83-9a2d0fd66931&ref_=pd_hp_d_btf_ci_mcx_mr_hp_d'
reviews = fetch_reviews(product_url, num_reviews=50)  # Adjust the number of reviews as needed

print(f"Number of reviews extracted: {len(reviews)}")


cleaned_reviews = [clean_text(review) for review in reviews]
print("Sample cleaned reviews:", cleaned_reviews[:3])  # Print first 3 cleaned reviews

all_cleaned_reviews = ' '.join(cleaned_reviews)
print(f"Combined cleaned reviews length: {len(all_cleaned_reviews)}")
print("Sample combined cleaned reviews:", all_cleaned_reviews[:200])  # Print first 200 characters

if all_cleaned_reviews.strip():  # Check if there is any non-whitespace character
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_cleaned_reviews)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
else:
    print("No data available to generate word cloud.")
#Fetch Reviews: Extracts and cleans reviews from the given product URL.
#Generate Word Cloud: Combines all cleaned reviews into a single string and generates a word cloud. Displays it using Matplotlib
# Sentiment analysis
sentiments = [analyze_sentiment(review) for review in reviews]
plt.hist(sentiments, bins=20, edgecolor='black')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Sentiment Distribution')
plt.show()
#Sentiment Analysis: Analyzes the sentiment of each review and plots a histogram of sentiment scores.
########################################################################
########################################################################
"""
6.	Write about the benefits/impact of the solution - in what way does the business (client) benefit from the solution provided?
"""
"""
----->

Enhanced Customer Insights:

Understanding Sentiments: Know if customers are happy or unhappy with products and services.
Targeted Feedback: Focus on what needs improvement based on positive and negative reviews.
Improved Product and Service Quality:

Product Development: Use feedback to refine and enhance products.
Service Enhancements: Identify service gaps and train staff accordingly.
Effective Marketing Strategies:

Brand Perception: Highlight positive feedback and address concerns in marketing.
Targeted Campaigns: Tailor marketing efforts based on common review themes.
Competitive Advantage:

Market Positioning: Adapt quickly to customer preferences to stay ahead of competitors.
Benchmarking: Compare feedback with competitors to find areas for improvement.
Operational Efficiency:

Automated Analysis: Save time with automated review processing and analysis.
Scalable Insights: Easily manage large volumes of data for ongoing insights.
Enhanced Customer Experience:

Personalized Responses: Tailor support based on customer feedback.
Proactive Issue Resolution: Address negative feedback promptly to improve customer satisfaction.
Strategic Decision Making:

Data-Driven Decisions: Make informed choices based on actual customer sentiments.
Trend Identification: Spot changes in customer preferences to adjust strategies.
Informed Product Launches:

Market Readiness: Assess how existing products are received to guide new launches.
Feature Validation: Confirm the need for new features based on existing feedback.
Risk Management:

Reputation Management: Monitor and address potential reputation risks early.
Crisis Management: Implement strategies quickly in response to emerging issues.
In essence, this solution helps businesses better understand and respond to customer needs, improve their offerings, create effective marketing strategies, and stay ahead of competitors.
"""


#######################################################################33

"""
Problem Statement: -
In the era of widespread internet use, it is necessary for businesses to understand what the consumers think of their products. If they can understand what the consumers like or dislike about their products, they can improve them and thereby increase their profits by keeping their customers happy. For this reason, they analyze the reviews of their products on websites such as Amazon or Snapdeal by using text mining and sentiment analysis techniques. 
"""
"""
Task 1:
1.	Extract reviews of any product from e-commerce website Amazon.
2.	Perform sentiment analysis on this extracted data and build a unigram and bigram word cloud. 
"""
#---->

import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.data import find
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob
from nltk import ngrams

# Download necessary NLTK data
try:
    find('corpora/stopwords.zip')
except LookupError:
    from nltk import download
    download('stopwords')

def fetch_amazon_reviews(product_url, num_reviews=10):
    reviews = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    while len(reviews) < num_reviews:
        try:
            response = requests.get(product_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            review_elements = soup.find_all('span', {'data-asin-review-id': True})
            for element in review_elements:
                if len(reviews) >= num_reviews:
                    break
                reviews.append(element.get_text())
            next_page = soup.find('li', {'class': 'a-last'})
            if next_page and next_page.a:
                product_url = f"https://www.amazon.in{next_page.a['href']}"
            else:
                break
        except requests.RequestException as e:
            print(f"Request error: {e}")
            break
    return reviews

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(tokens)

def generate_word_cloud(text, ngram=1):
    tokens = text.split()
    n_grams = ngrams(tokens, ngram)
    ngram_text = ' '.join(['_'.join(gram) for gram in n_grams])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(ngram_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Sentiment score between -1 and 1

# Task 1: Amazon Review Analysis
product_url = 'https://www.amazon.in/JBL-C100SI-Ear-Headphones-Black/dp/B01DEWVZ2C/?_encoding=UTF8&pd_rd_w=I1AmU&content-id=amzn1.sym.a490a4f5-1c2a-464e-8df8-c9593e3e7b44%3Aamzn1.symc.4d65739a-ff51-46fc-ad8f-587c12f6fb5e&pf_rd_p=a490a4f5-1c2a-464e-8df8-c9593e3e7b44&pf_rd_r=RAGE5E2YNFWFCMW1AXQG&pd_rd_wg=d4zrI&pd_rd_r=4b0c2b5f-d4e8-4f78-8f83-9a2d0fd66931&ref_=pd_hp_d_btf_ci_mcx_mr_hp_d'
reviews = fetch_amazon_reviews(product_url, num_reviews=50)
print(f"Number of Amazon reviews extracted: {len(reviews)}")

# Clean the reviews
cleaned_reviews = [clean_text(review) for review in reviews]
all_cleaned_reviews = ' '.join(cleaned_reviews)
print(f"Combined cleaned Amazon reviews length: {len(all_cleaned_reviews)}")
print("Sample combined cleaned reviews:", all_cleaned_reviews[:200])

# Generate unigram and bigram word clouds
if all_cleaned_reviews.strip():
    print("Generating unigram word cloud...")
    generate_word_cloud(all_cleaned_reviews, ngram=1)  # Unigram
    print("Generating bigram word cloud...")
    generate_word_cloud(all_cleaned_reviews, ngram=2)  # Bigram
else:
    print("No data available to generate word clouds.")

# Sentiment analysis for Amazon reviews
sentiments = [analyze_sentiment(review) for review in reviews]
plt.hist(sentiments, bins=20, edgecolor='black')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.title('Amazon Review Sentiment Distribution')
plt.show()
#############################################################


######################Task 2:
#1.	Extract reviews for any movie from IMDB and perform sentiment analysis.
                              ############################################
import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.data import find
from textblob import TextBlob
import matplotlib.pyplot as plt

# Download necessary NLTK data
try:
    find('corpora/stopwords.zip')
except LookupError:
    from nltk import download
    download('stopwords')

def fetch_imdb_reviews(movie_url, num_reviews=10):
    reviews = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    page_number = 1
    while len(reviews) < num_reviews:
        try:
            response = requests.get(f"{movie_url}?page={page_number}", headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            review_elements = soup.find_all('span', {'data-testid': 'review-text'})
            for element in review_elements:
                if len(reviews) >= num_reviews:
                    break
                reviews.append(element.get_text())
            page_number += 1
        except requests.RequestException as e:
            print(f"Request error: {e}")
            break
    return reviews

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(tokens)

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Sentiment score between -1 and 1

def plot_sentiment_histogram(sentiments):
    plt.hist(sentiments, bins=20, edgecolor='black')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.title('IMDB Review Sentiment Distribution')
    plt.show()

# IMDb URL for "Game of Thrones" reviews
movie_url = 'https://www.imdb.com/title/tt0944947/?ref_=nv_sr_srsg_0_tt_6_nm_2_in_0_q_game%2520'
reviews = fetch_imdb_reviews(movie_url, num_reviews=50)
print(f"Number of IMDB reviews extracted: {len(reviews)}")

# Clean the reviews
cleaned_reviews = [clean_text(review) for review in reviews]
all_cleaned_reviews = ' '.join(cleaned_reviews)
print(f"Combined cleaned IMDB reviews length: {len(all_cleaned_reviews)}")
print("Sample combined cleaned reviews:", all_cleaned_reviews[:200])

# Perform sentiment analysis
sentiments = [analyze_sentiment(review) for review in reviews]
plot_sentiment_histogram(sentiments)

# Print sentiment analysis results
print(f"Average Sentiment Score: {sum(sentiments) / len(sentiments)}")



##################################Task 3: 
#1.	Choose any other website on the internet and do some research on how to extract text and perform sentiment analysis
                                         ######
                                         
                                         
import requests
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.data import find
from textblob import TextBlob
import matplotlib.pyplot as plt

# Download necessary NLTK data
try:
    find('corpora/stopwords.zip')
except LookupError:
    from nltk import download
    download('stopwords')

def fetch_flipkart_reviews(product_url, num_reviews=10):
    reviews = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    page_number = 1
    while len(reviews) < num_reviews:
        try:
            response = requests.get(f"{product_url}?page={page_number}", headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract reviews
            review_elements = soup.find_all('div', {'class': '_16PBlm'})
            for element in review_elements:
                if len(reviews) >= num_reviews:
                    break
                review_text = element.get_text()
                reviews.append(review_text)
                
            # Check if there is a next page
            next_page = soup.find('a', {'class': '_1LKTO3'})
            if next_page and 'href' in next_page.attrs:
                next_page_url = next_page['href']
                product_url = f"https://www.flipkart.com{next_page_url}"
                page_number += 1
            else:
                break
        except requests.RequestException as e:
            print(f"Request error: {e}")
            break
    return reviews

def clean_text(text):
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    tokens = word_tokenize(text)  # Tokenize text
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]  # Apply stemming
    return ' '.join(tokens)

def analyze_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Sentiment score between -1 and 1

def plot_sentiment_histogram(sentiments):
    plt.hist(sentiments, bins=20, edgecolor='black')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.title('Flipkart Review Sentiment Distribution')
    plt.show()

# Flipkart URL for Realme phone reviews
product_url = 'https://www.flipkart.com/realme-p2-pro-5g-eagle-grey-128-gb/p/itm53d39fff9f20c?pid=MOBH45GSVQYKK5ME&param=8696&otracker=clp_bannerads_1_10.bannerAdCard.BANNERADS_Realme%2BP2%2BPro%2B5g%2BPL_mobile-phones-store_CIT1AQM8R81B'
reviews = fetch_flipkart_reviews(product_url, num_reviews=50)
print(f"Number of Flipkart reviews extracted: {len(reviews)}")

# Clean the reviews
cleaned_reviews = [clean_text(review) for review in reviews]
all_cleaned_reviews = ' '.join(cleaned_reviews)
print(f"Combined cleaned Flipkart reviews length: {len(all_cleaned_reviews)}")
print("Sample combined cleaned reviews:", all_cleaned_reviews[:200])

# Perform sentiment analysis
sentiments = [analyze_sentiment(review) for review in reviews]
plot_sentiment_histogram(sentiments)

# Print sentiment analysis results
print(f"Average Sentiment Score: {sum(sentiments) / len(sentiments)}")
#######################################################################