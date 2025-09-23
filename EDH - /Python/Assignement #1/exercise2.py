#a)
n = 0

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:   
            return False
    return True

#b)
def largest_prime_below(number):
    largest_square = None #creates the empty variable largest_square at the function squale
    
    #check all primes starting from 2 and rechecks everything from 2 to the <number
    prime = 2
    while prime * prime < number: #while the squarred prime is bellow our number
        if is_prime(prime):
            square = prime * prime
            largest_square = square  #keep updating to get the largest
        prime += 1
    
    return largest_square

#c)
while True:
    user_input = input("Please input a number (or 'q' to quit): ")
    
    if user_input =="q" :
        break

    #converts the input to a number so it raises no errors
    n = int(user_input)
    #finds the largest prime square below n now thats its an integer using our predefined function
    prime_square = largest_prime_below(n)
    
    if prime_square is not None:
        distance = n - prime_square #calculate the distance 
        print(f"The largest prime square below {n} is {prime_square}")
        print(f"Distance: {distance}")