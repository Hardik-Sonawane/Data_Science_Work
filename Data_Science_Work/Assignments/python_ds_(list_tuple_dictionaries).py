############## Assignment 2 ######################

# Que 1 
  # Write a program to reverse the order of list
# Solution ->
order_list = [1, 2, 3, 4, 5]
order_list.reverse() 
print("Reversed List:", order_list)


######################################################

# Que 2	Write a program to check number of
#       occurrences of specified elements in the list
# Solution ->
my_list = [1, 2, 3, 2, 4, 2, 5]
element = 2  
count = my_list.count(element)
print(f"Number of occurrences of {element}: {count}")

#############################################################

# Que 3 Write program to append the list1 
#        with list2 in the front
# Solution ->
list1 = [1, 2, 3]
list2 = [4, 5, 6]
list2.extend(list1) 
print("Combined List:", list2)

################################################################

# Que 4 Write program to insert new item before the second element
# Solution ->
my_list = [10, 20, 30, 40]
new_item = 99

my_list.insert(1, new_item)
print("Updated List:", my_list)

#############################################################

# Que 5 Write program to remove first occurrence 
#       of specified elements in the list,first you check 
#        which number is repeated and then remove it.
# Solution ->
my_list = [10, 20, 30, 20, 40, 20, 50]
print("Current List:", my_list)
element = int(input("Enter element to remove: "))

if element in my_list:
    my_list.remove(element)
    print("Updated list:", my_list)
else:
    print("Element not found")
   
##################################################################

# Que 6 write a program to check whether an element
#           exist in a tuple or not    
# Soution ->
my_tuple = (10, 20, 30, 40, 50)
element = int(input("Enter element to check: "))

if element in my_tuple:
    print(f"{element} exists in the tuple")
else:
    print(f"{element} does not exist in the tuple")
    
#####################################################################

# Que 7 write program to replace last value of each tuple
#       in list to 100  
# Solution ->
list_of_tuples = [(10, 20, 30), (40, 50, 60), (70, 80, 90)]

new_list = []
for t in list_of_tuples:
    new_tuple = t[:-1] + (100,) 
    new_list.append(new_tuple)

print("Modified list:", new_list)  

#############################################################

# Que 8 write a program to add a key and value
#        in the dictionary
# Solution ->
my_dict = {'a': 1, 'b': 2}
key = input("Enter key: ")
value = input("Enter value: ")
my_dict[key] = value
print("Updated Dictionary:", my_dict)

############################################################

# Que 9 #Write program to concatenate dictionary
# Solution ->
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}

dict1.update(dict2)
print("Concatenated Dictionary:", dict1)

############################################################3

# Que 10 #write program to create dictionary where
#        the keys are from 1 to 15 and values are from square of
#        the numbers
# Solution ->
square_dict = {}
for num in range(1, 16):
    square_dict[num] = num ** 2
print("Dictionary of squares:", square_dict)

#################################################################3

# Que 11  #Write a program to sum of all values in
#         the dictionary
# Solution -> 
my_dict = {'a': 100, 'b': 200, 'c': 300}
total = sum(my_dict.values())
print("Sum of all values:", total)

##################################################################
"""
Conclusion:
These Python programs helped me understand how to work
 with lists, tuples, and dictionaries.
 I learned different methods to manipulate data,
 check conditions, and perform calculations efficiently.
 """
 # End 