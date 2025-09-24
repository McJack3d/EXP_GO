#a)
n = 0 #create the input of the function is_prime and sets it a value of 0

def is_prime(n):
    if n < 2:
        return False #if number is strictly bellow 2 then not prime (2 is prime)
    for i in range(2, n):
        if n % i == 0: #for every value in between 2 and the input -1 checks if it can be divided by any of these value and leave a remainder of 0, if yes then it isn't prime
            return False
    return True #if both of the previous conditions aren't satisfied then the number is prime

#b)
def largest_prime_below(number):
    largest_square = None #creates the empty variable largest_square at the function scale
    
    #check all primes starting from 2 and rechecks everything from 2 to the <number
    prime = 2
    while prime * prime < number: #while the squarred prime is bellow our number
        if is_prime(prime): #checks if a number his prime
            square = prime * prime
            largest_square = square  #keep updating to get the largest
        prime += 1 #adds one to check every value
    return largest_square

#c)
while True:
    user_input = input("Please input a number (or 'q' to quit):")
    
    if user_input =="q" :
        break

    #converts the input to a number so it raises no errors
    n = int(user_input)
    #finds the largest prime square below n now thats its an integer using our predefined function
    prime_square = largest_prime_below(n)
    
    if prime_square is not None:
        distance = n - prime_square #calculate the distance 
        print(f"The largest prime square below {n} is {prime_square}")
        print(f"Distance:{distance}")