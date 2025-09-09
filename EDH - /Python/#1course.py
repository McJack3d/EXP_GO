"""
#prgm 1
print ("Hello Python !\
\nThis is my first programming code\
\nI am looking forward to write more ...")

#prgm 2
s_min = 60
min_h = 60
h_d = 24
d_w = 7
w = 5
number_sec_in5week = s_min * min_h * h_d * d_w * w
print ("The number of seconds in 5 weeks is " +str(number_sec_in5week))

#prgm 3
print ("What are you current savings")
savings = input ()
expected_savingsy1 = int(savings) * 1.0255
print ("Your expected savings at the end of the year : "+str(expected_savingsy1))

#prgm 4
print ("Whats the first number ?")
first_number = int(input ())
print ("Whats the second number ?")
second_number = int(input ())
sum_step1 = first_number + second_number
print ("The sum so far is " + str(sum_step1))
print ("Whats the third number ?")
third_number = int(input ())
sum_step2 = sum_step1 + third_number
print (f"The total sum is "+str(sum_step2))

#prgm 5
print ("Enter your First name : ")
firstname = input ()
print ("Enter your Last name :")
lastname = input ()
print (f"Your edhec email adress :\n"+(str(firstname)+str(lastname)).lower().replace(" ","")+"@edhec.com")

#prgm 6
print ("Whats the yearly demand for the product :")
ydp = int(input ())
print ("Whats the fixed ordering cost paid per order :")
foc = int(input ())
print ("Whats the annual holding cost per unit of product :")
ahc = int(input ())
ooq = ((2 * ydp * foc)/ahc) ** 0.5
print (f"The optimal order quantity is {ooq}")
"""

#prgm 7 (a)
first_value = True 
second_value = True
both_true = first_value and second_value

print(f"Both values are true : {both_true}")
#prgm 7(b)
third_value = False
three_true = both_true and third_value
print(f"The three booleans carry the same value : {three_true}")
 