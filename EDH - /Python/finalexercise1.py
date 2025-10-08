import pandas as pd
import numpy as np

def is_valid_loyalty_card(loyalty_card_number: int):
    card_str = str(loyalty_card_number)
    return len(card_str) == 6 and card_str[:2] == '48'
    
def determine_ticket_price(is_child: bool, has_loyalty_card: bool):
    if has_loyalty_card:
        return 4.99 if is_child else 8.99
    else:
        return 6.99 if is_child else 11.99

def main():
    total_group_price = 0.0
    customer_count = 0
    
    while True:
        customer_count += 1
        
        loyalty_input = int(input("Enter loyalty card number (-1 for no loyalty card): "))
        
        if loyalty_input == -1:
            has_loyalty_card = False
            print("No loyalty card.")
        else:
            if is_valid_loyalty_card(loyalty_input):
                has_loyalty_card = True
                print("Valid loyalty card detected.")
            else:
                print("Loyalty card number not valid, ignoring card.")
                has_loyalty_card = False
        
        child_input = input("Is child? (y/n): ").lower()
        is_child = child_input in ['y', 'yes']
        
        ticket_price = determine_ticket_price(is_child, has_loyalty_card)
        total_group_price += ticket_price
        
        more_customers = input("Are there more customers in the group? (y/n): ").lower()
        if more_customers in ['n', 'no']:
            print(f"Total customers:{customer_count}")
            print(f"Total group price: €{total_group_price:.2f}")
            return total_group_price

main()