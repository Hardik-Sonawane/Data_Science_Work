###############    Assignment 1    ######################

#  Que 1. write a program to check number is positive 
#       or negative or zero
NUM = int(input("Enter  a Number :"))
if NUM > 0:
    print("Number is Positive ")
elif NUM < 0 :
    print("Number is Negative ")
else :
    print(" Number is equal.")        
   
###############################################################

# Que 2.Write a program to check number is 
#       odd or even
NUMBER = int(input("Enter a Number"))
if NUMBER % 2 ==0 :
    print("Number is even ")
elif NUMBER % 2!= 0:
    print("Number is odd")
else :
    print("Number is not even or odd")    
    
#############################################################    

# Que 3.Given two non negative values print true
#       if they have same last digits
def same_digit(a,b):
    return str(a)[-1] == str(b)[-1]

first_number = int(input("Enter a non negative number"))
second_number = int(input("Enter a non negative number"))

print(same_digit(first_number,second_number))

#####################################################################
 
# Que 4.Write a program to print numbers 
#        from 1 to 10 with single row with one tab space
for i in range(1,11):
    print(i, end=" ")
    
##################################################################    

# Que 5.write a program to print even numbers between 23 
#       to 57.Each number should be printed in seperate row
for i in range(23,57):
    if i % 2 ==0:
        print("Even",i)
    else :
        print("Odd",i)

##############################################################

# Que 6. write program to get prime numbers

def prime_number(n):
    if n <=1:
        return False
    for i in range(2 ,int(n **0.5) +1):
        if n % 2 ==0:
            return False 
    return True    
Number = int(input("Enter a number"))    
if prime_number(Number):
    print(f'Number is Prime')
else:
    print("Number is not prime")    
    
##########################################################
# Que 7. Write program to write prime numbers between 
#        10 to 99
for i in range(10,100):
    if all(i % j !=0 for j in range(2, i)):
        print(i)

#########################################################

# Que 8.write program to calculate sum of all digits

No = int(input("Enter Number for sum "))
sum_of_no = sum(int(digit) for digit in str(No))
print(sum_of_no)

########################################################

# Que 9. Write program to reverse the number
NUM = (input("Enter a Number "))
print(NUM[::-1])

#############################################################

# Que 10. Write program to check number is palindrome
Pal1 = input("Enter first string")
if (Pal1 == Pal1[::-1]):
    print("Palindrome")
else:
    print("Not Palindrome")    

##########################################################





