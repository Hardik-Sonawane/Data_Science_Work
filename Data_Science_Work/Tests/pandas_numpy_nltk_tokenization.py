# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 09:02:26 2024

@author: scs
"""  ######### Test 3 ####################3

## Q1.1. Write a Pandas program to convert Series of lists to one Series.
#Sample Output: 
#Original Series of list
#0 [Red, Green, White]
#1 [Red, Black]
#2 [Yellow]
#dtype: object
#One Series
#0 Red
#1 Green
#2 White
#3 Red
#4 Black
#5 Yellow
#dtype: object
# Answer:

import pandas as pd
a=pd.Series([['Red', 'Green', 'White'], ['Red', 'Black'], ['Yellow']])
print(a)
a1= pd.Series([item for sublist in a for item in sublist])
print(a1)

##################################################

#Q3. Create a result array by adding the following two NumPy arrays. Next, 
#modify the result array by calculating the square of each element
#arrayOne = numpy.array([[5, 6, 9], [21 ,18, 27]])
#arrayTwo = numpy.array([[15 ,33, 24], [4 ,7, 1]]
# Answer:
    
import pandas as pd 
import numpy as np
arrayOne = np.array([[5, 6, 9], [21, 18, 27]])
arrayTwo = np.array([[15, 33, 24], [4, 7, 1]])
b = arrayOne + arrayTwo
print(b)
c=np.square(b)
print(c)

###################################################


#Q2. Write a python NLTK program to split the text sentence/paragraph into 
#a list of words.
#text = '''
#Joe waited for the train. The train was late. 
#Mary and Samantha took the bus. 
#I looked for Mary and Samantha at the bus station.
#'''
#Answer:
import nltk
from nltk.tokenize import word_tokenize
text = '''
Joe waited for the train. The train was late. 
Mary and Samantha took the bus. 
I looked for Mary and Samantha at the bus station.
'''
nltk.download('punkt')
d=word_tokenize(text)
print(d)    
    
#################################################3
#Q4. Write a python program to extract word mention someone in tweets 
#using @ from the specified column of a given DataFrame.
#DataFrame: ({
 #'tweets': ['@Obama says goodbye','Retweets for @cash','A political endorsement in 
#@Indonesia', '1 dog = many #retweets', 'Just a simple #egg']
 #})
#Answer:
    
    ##############################################
    
    
#Q5. Write a NumPy program to compute the mean, standard deviation, and 
#variance of a given array along the second axis.
#array:
#[0 1 2 3 8 5
## Answer:
    
    
    #############################################