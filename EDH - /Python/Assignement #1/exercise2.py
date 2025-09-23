#a)
n = int(input("Please input a number: "))

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):  # on teste tous les nombres de 2 jusqu'à n-1
        if n % i == 0:     # si on trouve un diviseur
            return False   # ce n'est pas premier
    return True            # sinon, c'est premier

#b)
def largest_prime_below(number):
    largest_prime_square = 0
    
    # Check all numbers from 2 up to sqrt(number)
    for i in range(2, int(number ** 0.5) + 1):
        if is_prime(i):  # if i is prime
            prime_square = i * i  # calculate its square
            if prime_square < number:  # if square is below our number
                largest_prime_square = prime_square  # update largest
    
    return largest_prime_square

#c) Main code
while True:
    user_input = input("Please input a number (or 'q' to quit): ")
    
    if user_input == "q":
        break
    
    try:
        n = int(user_input)
        if n <= 1:
            print("Please enter a number greater than 1")
            continue
        
        largest_square = largest_prime_below(n)
        if largest_square == 0:
            print(f"No prime square found below {n}")
        else:
            distance = n - largest_square
            print(f"Distance from {n} to largest prime square below it: {distance}")
    
    except ValueError:
        print("Please enter a valid number or 'q'")