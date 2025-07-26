
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