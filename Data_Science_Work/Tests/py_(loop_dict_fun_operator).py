# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 08:19:53 2025

@author: maith
"""

#1.	Write a program to print numbers from 1 to 10 with single row with one tab space

for i in range(1,11):
    print(i,end=" ")
    
#2.	Write program to create dictionary where the keys arefrom 1 to 15 and values are from square of the numbers

dict1={num:num**2 for num in range(1,16)}
print(dict1)

#3. Find out number is prime or not
def isprime():
    num=int(input("Enter num:"))
    if num > 1:  
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                print("Number is not prime")
                break
        else:
            print("Number is prime")
isprime()

#4. Write the function which will measure number of vowels in sentence

def count_vowels(str1):
    count=0
    new_str=list(str1)
    for i in new_str:
        if i=='a' or i=='A' or i=='e' or i=='E' or i=='o' or i=='O' or i=='i' or i=='I' or i=='u' or i=='U':
            count+=1
            
        else:
            None
    return count        
str1=input("Enter a sentence: ")
print(count_vowels(str1))

#5. Find out gcd of two number

def get_gcd(num1,num2):
    while(num2):
       num1,num2=num2,num1%num2
    
    return num1

num1=int(input("Enter the number 1: "))
num2=int(input("Enter the number 2: "))
gcd=get_gcd(num1,num2)
print(f'GCD of {num1} , {num2} is {gcd}')

#10 to 100 prime
def isprime():
    for num in range (10,101):
    
        if num > 1:  
            for j in range(2, int(num ** 0.5) + 1):
                if num % j == 0:
                    print(f"{num} is not prime")
                    break
            else:
                print(f"{num} is prime")
isprime()