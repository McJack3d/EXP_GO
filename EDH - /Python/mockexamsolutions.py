import pandas as pd
import numpy as np

dic = {"A" : 0.99, "B" : 1.99, "C" : 4.99, "D" : 9.99}

def compute_price(category : str) :
    return dic[category]

def compute_sales_price(category : str, taxRate : float):
    taxrate = 1 + taxRate
    sales_price = taxrate * compute_price(category)
    return sales_price

# Main user interface
def main():
    sold_items = 0
    
    while True:
        category = input("Category of the item ('q' to end): ")
        
        if category == "q":
            break
        
        # Check if category is valid
        if category not in dic:
            print("Invalid category. Skipping this product.")
            continue
        
        taxRate = float(input("Tax rate (in decimal format): "))
        
        # Calculate and display sales price
        sales_price = compute_sales_price(category, taxRate)
        print(f"The sales price is: {round(sales_price, 2):.2f}")
        
        sold_items += 1
    
    print(f"The number of sold items: {sold_items}")

# Run the main function
if __name__ == "__main__":
    main()