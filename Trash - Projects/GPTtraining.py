"""
name = str(input("Please enter your name: "))

print(f"Hello {name} and welcome to the world of coding !")
---
n = int(input("Please input an integer number: "))

if n%2 == 0 :
    result = False
else : 
    result = True

print (result)
---
print("Voici le compteur de voyelles")

word = str(input("Please input the word for which you want to count the number of vowels: ")).lower()
wordlength = len(word)
vowels = ["a", "e", "i", "o", "u", "y"]
count = 0

for i in range(wordlength) :
    if word[i] in vowels :
        count += 1

print(count)

---

shopping_list = []
n = ""

while True : 
    n = str(input("Please input the next item you'd like to add to the shopping list or 'q' to quit:"))
    if n != "q" :
        shopping_list.append(n)
    else:
        print(shopping_list)
        break
"""
