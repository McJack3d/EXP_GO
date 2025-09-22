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
"""
#prgm 2
print("Please enter each number of the serie you want to see averaged(enter q when you wanna quit) : ")
x = (input("Which number to add : "))

while x != "q" :
    x = (input("Which number to add : "))
    x = x + x
    if x = 
else :
    print(f"The average of your serie is {x}")