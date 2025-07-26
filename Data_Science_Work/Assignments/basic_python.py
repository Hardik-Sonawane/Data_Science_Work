#write a program to check number is positive or negative or zero
num=int(input("Enter the Value of number: "))
if num<0:
    print("Number is Negative!!")
elif num>0:
    print("Number is Positive!!")
else:
    print("Number is Zero")
    
#Write a program to check number is odd or even
num=int(input("Enter the value: "))
if num%2==0:
    print("Number is Even.")
else:
    print("Number is odd.")    

#Given two non negative values print true if they have same last digits
num=int(input("Enter value of first number: "))
num1=int(input("Enter value of second number: "))
a=num%10
b=num1%10
if num>0 and num1>0:
    print(a==b)
else:
    print("One of the number or both numbers are negative.")
#Write a program to print numbers from 1 to 10 with single row with one tab space
for i in range(1,11):
    print(i,end=' ')

#write a program to print even numbers between 23 to 57.Each number should be printed in seperate row
for i in range(23,58):
    if i%2==0:
        print(i)
        
#write program to get prime numbers
def get_numbers(From,Limit):
    for i in range(From,Limit+1):
        if i<=1:
            print(False)
        else:
            is_prime=True
            for j in range (2,int(i**0.5)+1):
                if i%j==0:
                    is_prime=False
                    break
            print(f'{i}: is prime: {is_prime}')
From=int(input("Enter from where you want prime numbers: "))
Limit=int(input("Enter upto where you want prime numbers: "))
get_numbers(From,Limit)
 
#Write program to write prime numbers between 10 to 99
for i in range(11,99):
    if i<=1:
        print(False)
    else:
        is_prime=True
        for j in range(2,int(i**0.5)+1):
            if i%j==0:
                is_prime=False
                break;
        print(f'{i} is prime:{is_prime}')        
        
#write program to calculate sum of all digits
num=int(input("Enter the number: "))
a=num
total=0
while a!=0:
     b=a%10
     total+=b
     a=a//10
print(f'The sum of {num} is {total}')
    
#Write program to reverse the number
num=int(input("Enter the number: "))
a=num
reverse=0
while a!=0:
     reverse=(10*reverse)+(a%10)
     a=a//10
print(f'The reverse of {num} is {reverse}')


#Write program to check number is palindrome
num=int(input("Enter the number: "))
a=num
reverse=0
while a!=0:
     reverse=(10*reverse)+(a%10)
     a=a//10
if num==reverse:
    print(f"{num} is Palindrome")
else:
    print(f'{num} is not an palindrome')
 
