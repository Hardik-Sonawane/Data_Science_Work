#1.Write a program to reverse the order of list
list1=["Rose","Mogara","Sunflower"]
list1.reverse()
print("Reverse list:",list1)

#2.Write a program to check nember of occurences of specified element in the list
list1=[1,2,1,1,3]
n=list1.count(1)
print(n)

#3.Write program to append the list1 with list2 in front
list1=["Vivo","OnePlus","Samsung"]
list2=["Moto","Oppo","Lenevo"]
list1.append(list2)
print(list1)

#4.Write program to insert new item before the second element
list1=["Parrot","Sparrow"]
list1.insert(1,"Pigeon")
print(list1)

#5.Write program to remove first occurrence of specified elements in the list,first you check which number is repeated and then remove it.
list1 = [1, 2, 3, 1, 4, 1]

for num in list1:
    if list1.count(num) > 1:  
        list1.remove(num)  
        break  

print(list1)

#6.Write a program to check whether an element exits in a tuple or not
tup = (1, 2, 3, 4, 5)
element = 3

if element in tup:
    print(f"{element} exists in the tuple.")
else:
    print(f"{element} does not exist in the tuple.")

#8.write a program to add a key and value in the dictionary
dict1 = {'a': 1, 'b': 2}
dict1['c'] = 3

print(dict1)


#9.Write program to concatenate dictionary
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}

dict1.update(dict2)

print(dict1)


#10.write program to create dictionary where the keys are from 1 to 15 and values are from square of the numbers
squares_dict = {num: num**2 for num in range(1, 16)}

print(squares_dict)


#11.Write a program to sum of all values in the dictionary
dict1 = {'a': 10, 'b': 20, 'c': 30}
total = sum(dict1.values())

print("Sum of values:", total)
