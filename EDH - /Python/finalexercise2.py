import numpy as np 
import pandas as pd

def number_d_similar(first_list: list,second_list: list, distance: int):
    min_length = min(len(first_list), len(second_list))
    similar_pairs = 0
    
    for i in range(min_length):
        if abs(first_list[i] -second_list[i]) <= distance:
            similar_pairs += 1
    
    return similar_pairs

def determine_similarity_score(first_list: list,second_list: list):
    distances = {}
    
    for i in range(len(first_list)):
        d = abs(first_list[i] - second_list[i])
        distances[i] = d
    
    sorted_distances =sorted(distances.values())
    target_index = len(first_list)// 2
    
    return sorted_distances[target_index]

list1= [8, 10, -29,45]
list2 = [6,14, -26, 39]

print(f"Dist2:{number_d_similar(list1, list2, 2)}pairs")
print(f"Dist4: {number_d_similar(list1,list2, 4)}pairs") 

score = determine_similarity_score(list1, list2)
print(score)

