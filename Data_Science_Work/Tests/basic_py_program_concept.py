###1
20,4

###2
fruits=["Strawberries","Nectarines","Apples","Grapes","Peaches","Cherries","Pears"]
for fruit in fruits:
    if fruit == "Apples":
        print(fruit)
    

#######3
fruits=["Strawberries","Nectarines","Apples","Grapes","Peaches","Cherries","Pears"]
fruits[-1]="Melons"
fruits.append("Lemons")
print(fruits)


#############4
starting_dictionary = {
    "a": 9,
    "b": 8,
}
final_dictionary = {
    "a": 9,
    "b": 8,
    "c": 7,
}
starting_dictionary.update(final_dictionary)
print(starting_dictionary)

    



############5
order = {"starter": {1: "Salad", 2: "Soup"},
	    "main": {1: ["Burger", "Fries"], 2: ["Steak"]},
	    "dessert": {1: ["Ice Cream"], 2: []},}
print(order)


########6
def add(n1, n2):
	  return n1 + n2
def subtract(n1, n2):
	  return n1 - n2
def multiply(n1, n2):
	  return n1 * n2
	 
def divide(n1, n2):
	  return n1 / n2
print(add(2, multiply(5, divide(8, 4))))


#########7
def outer_function(a, b): 
   def inner_function(c, d): 
    return c + d
   return inner_function(a, b)	 
result = outer_function(5, 10)
print(result)

###########13
Func1=lambda x: ((x + 3) * 5 / 2)
Func1(3)

#######12.
lst= [ int(x/3) for x in range(5,35) if x % 2 == 0 and x%5 == 0]
print(lst[-1]+5)


##########11
lst = [int(x*x) for x in range(3,12,4)]
print(lst[-2])

########

student = {
  "name": "Emma",
  "class": 9,
  "marks": 75
}
marks_value = student["marks"]
print(marks_value)

################
student = {
  "name": "Emma",
  "class": 9,
  "marks": 75
}
student.pop("marks")

##################
dict1=["2","3"]
dict2=dict1.copy()
print(dict2)
########################

