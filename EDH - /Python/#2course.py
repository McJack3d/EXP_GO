"""
#prgm 1
print ("At what speed is the car going ?")
speed_car = float(input())
speed_limit = 80
if speed_car >= speed_limit :
    print ("Car is speeding !")

#prgm 2
print ("What's a stock you've invested in the current year ?")
stock_name = input()
print ("What as been the stock return over the last year in percent ?")
stock_return = float(input())
savings_return = 2.3

if stock_return > savings_return :
    print ("Nice your stock performed better than a saving account !")
else:
    print ("You might reconsider your investment towards a savings account !")

#prgm 3
print ("Please input your postal code :")
postal_code = str(input())
if postal_code[:2] == "59" :
    print ("The postcode is in the Nord department.")
else :
    print ("The postcode is not in the Nord department.")

#prgm 4
print ("Which day of the week do you want to translate ?")
day = str(input().lower())
print ("The name in Latin is :")
if day == "monday" or day == "lundi" : 
    print ("Dies Lunae")
elif day == "tuesday" or day == "mardi" :
    print ("Dies Martis")
elif day == "wednesday" or day == "mercredi" :
    print ("Dies Mercurii")
elif day == "thursday" or day == "jeudi" :
    print ("Dies Iovis")
elif day == "friday" or day == "vendredi" :
    print ("Dies Veneris")
elif day == "saturday" or day == "samedi" :
    print ("Dies Saturni")
elif day == "sunday" or day == "dimanche":
    print ("Dies Solis")  

#prgm 5
print ("Do you reside in mainland France ?(y/n)")
residence = str(input().lower())
print ("Have you made more than 2 claims in the last 10 years ?(y/n)")
claims = str(input().lower())
print ("Are you resgistered as a bad payer within the corresponding national registry ?(y/n)")
bad_payer = str(input().lower())
if residence == "y" and claims == "n" and bad_payer == "n":
    print ("You are welcome to register to our insurance company !") 
else : 
    print ("Unfortunatly, your condition do not allow your to resgiter to our insurance company...")
    
#prgm 6
print ("How old are you ?")
age = int(input())
print ("Where do you live ?(roubaix/mel/elsewhere)")
house = str(input())

if age <= 12:
    fare = 0
elif 13 <= age <= 17 and house == "roubaix":
    fare = 3.50
elif 13 <= age <= 17:
    fare = 4.50
elif 18 <= age < 64:
    if house == "roubaix":
        fare = 5.50
    elif house == "mel":
        fare = 6.00
    else:
        fare = 8.00
elif age >= 65:
    print("Are you entitled to reductions? (y/n)")
    reduc = input().strip().lower()
    if reduc == "y":
        fare = 4.50
    else:
        fare = 5.50

print(f"You'll have to pay {fare}€")

#prgm 7
#a
print ("How many numbers do you want to print ?")
numbers = int(input())

for i in range(numbers + 1):
    print(i)

#b
print ("How many numbers do you want to print ? (please enter a strictly positive number)")
numbers = int(input())
while numbers < 0 :
    print ("Invalid selection please enter a positive number.")
    print ("How many numbers do you want to print ? (please enter a strictly positive number)")
    numbers = int(input())

print ("What should the starting number be ? (please enter a strictly positive number)")
starting_numbers = int(input())
while starting_numbers < 0 :
    print ("Invalid selection please enter a positive number.")
    print ("What should the starting number be ? (please enter a strictly positive number)")
    starting_numbers = int(input())

for i in range(starting_numbers, numbers + 1):
    print(i)

#prgm 8
print("What is the current year ?")
current_year = int(input())
starting_year = current_year - 4 
px = 0
for i in range(starting_year, current_year):
    print (f"What was the profit in {i}?")
    px = px + int(input())

print(f"Total profit over the last 4 years :{px}")
"""

#prgm 9
print ("Wich Fibonacci number would you want to ")