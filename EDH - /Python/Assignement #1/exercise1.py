#a)
salary_grade_A = {
    "A" : 2700,
    "B" : 2900,
    "C" : 3300,
    "D" : 3600,
    "E" : 3900
}

salary_grade_B = {
    "A" : 3800,
    "B" : 4200,
    "C" : 4400,
    "D" : 4700,
    "E" : 5100
}

def salary_grade(years_similar, years_lower):
    years_exp = years_lower/2 + years_similar
    
    if years_exp <= 2:
        grade_letter = "A"
    elif years_exp <= 4:
        grade_letter = "B"
    elif years_exp <= 6:
        grade_letter = "C"
    elif years_exp <= 8:
        grade_letter = "D"
    else:
        grade_letter = "E"
    
    return grade_letter

#b)
def minimum_salary(role, grade):
    if role == "analyst":
        return salary_grade_A[grade]
    else:
        return salary_grade_B[grade]

#c)
def is_above_minimum(role, grade, salary):
    min_salary = minimum_salary(role, grade)
    return salary >= min_salary

#d) Main program
employee_type = input("Are you calculating for an existing or new employee? (existing/new): ")

if employee_type == "new":
    years_similar = int(input("Years of experience at similar level: "))
    years_lower = int(input("Years of experience at lower level: "))
    role = input("Intended role (analyst/senior analyst): ")
    
    grade = salary_grade(years_similar, years_lower)
    min_sal = minimum_salary(role, grade)
    
    print("Employee Grade:", grade)
    print("Minimum salary for", role, "at grade", grade, ": $", min_sal)
    
elif employee_type == "existing":
    role = input("Employee role (analyst/senior analyst): ").lower()
    grade = input("Employee grade (A/B/C/D/E): ").upper()
    current_salary = int(input("Current monthly salary: $"))
    
    min_sal = minimum_salary(role, grade)
    is_above = is_above_minimum(role, grade, current_salary)
    
    print("Minimum salary for", role, "at grade", grade, ": $", min_sal)
    print("Current salary: $", current_salary)
    
    if is_above:
        print("Employee is at or above minimum salary requirement")
    else:
        print("Employee is below minimum salary requirement")
        
else:
    print("Please enter 'existing' or 'new'")