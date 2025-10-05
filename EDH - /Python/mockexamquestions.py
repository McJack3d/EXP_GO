import pandas as pd
import numpy as np
"""
dic = {"A" : 0.99, "B" : 1.99, "C" : 4.99, "D" : 9.99}
taxRate = 0.0

def compute_price(category : str) :
    return dic[category]

def compute_sales_price (category : str, taxRate : float):
    sales_price_bt = compute_price(category)
    sales_price = sales_price_bt + sales_price_bt * taxRate
    return sales_price


while True :
    category = str(input("Please input the item Category letter(A,B,C or D) or 'q' to quit: "))
    if category == "q":
        break
    else :
        taxRate = str(input("Please input the item Category letter(A,B,C or D) or 'q' to quit: "))
        print(f"The sales price for this item will be {compute_sales_price(category, taxRate)}")
"""

def first_woodall_numbers(k : int):
    