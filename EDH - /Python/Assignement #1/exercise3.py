print("Welcome to the word similaraty comparator")
first_word = str(input("Please enter a word: "))
second_word = str(input("Please enter the word to compare it: "))

def similarity_score(first_word, second_word):
    smallest_word = min(len(first_word), len(second_word))
    baseline = 0

    #computation of the similarity using the smallest word as reference
    for i in range(smallest_word):
        if first_word[i]==second_word[i]:
            baseline += 1
    
    #computation of the score
    max_length = max(len(first_word), len(second_word))
    if max_length == 0:
        return 100  # Both words are empty
    
    similarity_percentage = (baseline / max_length) * 100
    return similarity_percentage

# Call the function with the input words
result = similarity_score(first_word, second_word)
print(f"Similarity score: {result:.1f}%")
