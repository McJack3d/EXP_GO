"""
#prgm 1
k = int(input("Wich base number k do you wanna see the largest computed power : "))

u = int(input("Which upper bound should we apply it to(stictly more than base number k) :"))

while k >= u :
    print ("Error, the base number k cannot be higher than the upper bound")
    k = int(input("Wich base number k do you wanna see the largest computed power : "))

    u = int(input("Which upper bound should we apply it to(stictly more than base number k) : "))

power = 1

while power * k < u :
    power *= k

print(f"The largest power of {k} strictly below {u} is {power}")

#prgm 2
print("Please enter each number of the serie you want to see averaged(enter q when you wanna quit) : ")
x = (input("Which number to add : "))

while x != "q" :
    x = (input("Which number to add : "))
    x = x + x
    if x = 
else :
    print(f"The average of your serie is {x}")

#prgm 3
length = float(input("Input the length of the rectangle :"))
width = float(input("Input the width of the rectangle :"))

def rectangle_area(length, width):
    area = length * width
    return area

print(rectangle_area(length, width))

#prgm 4
savings = float(input("Input your actual saving : "))
expected_rr = float(input("Input the expected interest rate for the next year (in %): "))

def compute_savings(savings, expected_rr):
    savings_y1 = savings * (expected_rr/100) +savings
    return savings_y1

print(f"The expected total savings after year 1 should be : {compute_savings(savings, expected_rr)} ")
"""
#prgm 5
#a)
base = int(input("Input the number you wanna see the factorial of :"))

def compute_factorial(base) : 
    for base in base :
     n =+ base 
    return n  

print(compute_factorial(base))