print("\nWelcome to the word similaraty comparator\n")
first_word = str(input("Please enter a word: "))
second_word = str(input("Please enter the word to compare it: "))

def similarity_score(first_word, second_word):
    small_word = min(len(first_word), len(second_word))
    baseline = 0

    #computation of the similarity using the smallest word as reference
    for i in range(small_word):
        if first_word[i]==second_word[i]:
            baseline += 1
    
    #computation of the score
    max_length = max(len(first_word), len(second_word))

    similarity = (baseline / max_length) * 100
    return similarity

#uses the function
result = similarity_score(first_word, second_word)
#prints the result
print(f"\nSimilarity score:{result}%")
