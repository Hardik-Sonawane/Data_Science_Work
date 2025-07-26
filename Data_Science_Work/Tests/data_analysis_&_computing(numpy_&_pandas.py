########## Test-4 16/04/25   ######################

# Que.1 Write a Numpy Program to create an element wise comparison
#       (greater,greater_equal,less and less_equal) of two given arrays
# Solution:-
import numpy as np 
f1 = np.array([2,3,4])
f2 = np.array([5,6,7])
# first 
print("The greater is :",np.greater(f1,f2))
#second
print("The greater_equal is:",np.greater_equal(f1,f2))
# Third
print("The less one is:",np.less(f1,f2))
# Fourth
print("The less_equal is :",np.less(f1,f2))

#######################################################

# Que 2.You are given the following data about employees in a company
#       data={ 'Name':['Alice','Bob','Charlie','David','Eva'],'Age':[25,30,35,45,28],
#          'Department': ['HR','IT','Finanance','IT','HR'],
#           'JOining_year':[2018,2016,2015,2019,2020]}
# Create a DataFrame from the aboe dictionary and display the following
# the first two rows , the column_names,data typs of each colum,summary statistics for numerc column
# Solution:-
import pandas as pd
import numpy as np
data={ 'Name':['Alice','Bob','Charlie','David','Eva'],
       'Age':[25,30,35,45,28],
       'Department': ['HR','IT','Finanance','IT','HR'],
       'Joining_year':[2018,2016,2015,2019,2020]}
df=pd.DataFrame(data)

# first
print("first two rows",df.head(2))
#second
print("the olumn names",df.columns)
# Third
print("the data types is :",df.dtypes)
# Fourth
print("Summary statistics for numeric column",df.describe())

###################################################

# Que 3.Extend the DataFrame from Question 2 
#   Add a new column called Salary with Values:[50000,60000,70000,65000,48000].
#   Calculate a new column Experience as 2025- joining _year.
#   Create a new column Seniority
#   'Junior' if experience <5 years
#    'Mid' if 5<= experience <= 8 years 
#     'Senior' if experience >= 8 years
# Solution :
    
import pandas as pd 
import numpy as np 
data={ 'Name':['Alice','Bob','Charlie','David','Eva'],
       'Age':[25,30,35,45,28],
       'Department': ['HR','IT','Finanance','IT','HR'],
       'Joining_year':[2018,2016,2015,2019,2020]
       }

df=pd.DataFrame(data)

df['salary']=[50000,60000,70000,65000,48000]
df['Experience']= 2025 - df['Joining_year']
# labels=['Junior','Mid','Senior']
df['Seniority']='Junior'
df.loc[df['Experience'] >= 5, 'Seniority'] ='Mid'
df.loc[df['Experience'] >=8, 'Seniority']='Senior'
print(df)

###############################################################

# Que 4. Write a Numpy program to complete the multipliplication of two given marixes
# p=[[1,0],[0,1]]
# q = [[4,2],[1,3]]
# Solution : 
import pandas as pd 
import numpy as np 
p=[[1,0],[0,1]]
q = [[4,2],[1,3]]
print("Multiplication is :",np.multiply(p,q))

###############################################################
################################################################
