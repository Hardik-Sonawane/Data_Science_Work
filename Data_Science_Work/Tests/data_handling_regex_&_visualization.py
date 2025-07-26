##############  All Test   #############################
#que 1. Write a function that takes a list of integers and returns a 
#new list with only the even numbers, sorted in descending order
# Solution :
def ret_even(number):
    return sorted([n for n in number if n% 2 ==0],reverse =True )

########################################################

#Que 2.Write a program that takes a 
#user's full name as input and outputs their
 #initials in uppercase (e.g., Virat Kohli → V.K.).
# Solution:
def initial_letter(full_name):
    a = full_name.strip().split()
    initials = ""
    for f in a:
        if f:
            initials += f[0].upper() + "."
    return initials

name = input("Enter  name: ")
print(initial_letter(name))

##########################################################

# Que.4 Write a script that reads a .txt file and counts
# the number of lines, words, and characters.
# Solution:

    





# 6. Given a list of dictionaries with 
#student info ({'name': ..., 'score': ...}),
# return names of students with score > 75.
# solution:
students =[{'name':'ram','score':70},
          {'name':'shyam','score':90}]
a=[student['name'] for student in students if student['score']>75]
print(a)
    

########################################################

#Que.7  Given a list of tuples (name, age),
# sort them by age using a lambda function.
#Solution:
a=[('ram',40),('shyam',20),('dham',25),('hari',50)]
a.sort(key=lambda x: x[1])
print(a)
   
#######################################################

# Que.11 Plot a bar chart showing average scores of students 
# per subject from a given DataFrame.
# data = {'Math': [90, 85, 88], 'Science': [80, 82, 84], 'English': [78, 75, 80]}
#Solution:
import matplotlib.pyplot as plt
import pandas as pd 

data = {'Math': [90, 85, 88],
        'Science': [80, 82, 84],
        'English': [78, 75, 80]}

df=pd.DataFrame(data)
average=df.mean()
average.plot(kind='bar', color='red')
plt.title('average score of student')  
plt.xlabel('subject')
plt.ylabel('score')
plt.tight_layout()
plt.show()
########################################################

#  Que 9. Load a employee.csv file, filter rows where column
# "salary" is above the median, and show the top 5.
#Solution:
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv('E:/data-science/re_3-statistics/employees.csv')
a=df['salary'].median()
medium=df[df['salary'] > a]
print(medium.head(5))

##############################################################

# Que. 15.Replace all digits in a string with #.
# text = "My phone number is 1234567890."
import re
text = "My phone number is 1234567890."
a = re.sub(r'\d', '#', text)
print(a)
#####################################################
# Que. 13. Given a text, extract all valid email
#addresses using regex.
#text = "For more details, Please contact us at
# support@sanjivani.com or admin-info@sanjivani.co.ind."
# Solution:
text=re.findall()



########################################################################
    
# Que.10 Load a data.csv file with missing values.
# Show how to fill missing numerical values with the column mean.
#Solution:
import pandas as pd
df = pd.read_csv('E:/data-science/re_3-statistics/data.csv')
df_filled = df.fillna(df.mean(numeric_only=True))
print(df_filled)
##########################################################

#Que.3. Write a function to generate
# the first n Fibonacci numbers using a loop.
# Solution:
    
def fibonacci(n):
    sequence = []
    a,b =0,1
    for _ in range(n):
        sequence.append(a)
        a,b = b,a+b
    return sequence
print(fibonacci(5))    
#####################################################
   
    
   
    
   
    
   