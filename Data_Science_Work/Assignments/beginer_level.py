
#Exercise1
def mailing_address():
    name="Sonawane Hardik"
    Address="At-Lakhefal Tal-Shevgaon \n Dist-Ahmednagar"
    State="Maharashtra"
    email="sonawanehardik080@gmail.com"
    pin_code="Pin-414502"
    Country="India"
    print(f"{name}\n{Address}\n{State}\n{email},\n {pin_code}\n{Country}")
    
mailing_address()    


#Area of field
width=float(input("enter width of field: "))
length=float(input("enter length of field:"))
area_square_feet=length*width
area_acres=area_square_feet/43560
print("area of the field is {:.2f} acres".format(area_acres))



#Area of room 
width=float(input("enter width of room: "))
length=float(input("enter length of room:"))
area=width*length
print(f" area of room is :{area} square feet")




#bottle deposites
num_small_cont = int(input("Enter the number of small containers (holding one liter or less): "))
num_large_cont = int(input("Enter the number of large containers (holding more than one liter): "))

refund_small=num_small_cont*0.10
refund_large=num_small_cont*0.25
total_refund=refund_small+refund_large
    
print(f"Refund amount is:${total_refund:.2f}")




# tax and tip
m_cost=float(input("enter the cost of the meal:"))
tax_rate=0.08
tip_rate=0.18

tax_amount=m_cost*tax_rate
tip_amount=m_cost*tip_rate
total_cost=m_cost+tax_amount+tip_amount

print(f" tax amount is: {tax_amount:.2f}")
print(f" tip amount is : {tip_amount:.2f}")
print(f"total is : {total_cost:.2f}")


#Height Units

def feet_and_inches_to_cm():
    feet = float(input("Enter the number of feet: "))
    inches = float(input("Enter the number of inches: "))

    total_inches = feet * 12 + inches
    cm = total_inches * 2.54

    print("The equivalent height in centimeters is:", cm)

feet_and_inches_to_cm()













