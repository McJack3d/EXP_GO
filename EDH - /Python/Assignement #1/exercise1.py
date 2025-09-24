#a)
years_similar = 0
years_lower = 0
years_total = 0
grade = "" #creates a bunch of values that will use later and asigns them values so it raises no errors

def salary_grade(years_similar, years_lower): #defines the salary_grade function
    years_total = years_lower * 0.5 + years_similar #calculates the total senority using both variables
    if years_total < 2 : #for each slices assigns the string value of a grade
        grade = "A"
    elif 2 <= years_total <= 5 :
        grade = "B"
    elif 6 <= years_total <= 11 :
        grade = "C"
    elif 12 <= years_total <= 19 :
        grade = "D"
    elif years_total >= 20 :
        grade = "E"
    return grade    #returns the grade

#b)
role = ""
referenceA= { "A" : 2700, "B" : 2900, "C" : 3300, "D" : 3600, "E" : 3900}
referenceS= { "A" : 3800, "B" : 4200, "C" : 4400, "D" : 4700, "E" : 5100} #creates a dictionnary for each role analyst and senior that uses grades as keys and salaries as values

def minimum_salary(role, grade): #defines the minimum_salary function based on grade and role

    if role == "analyst" : #checks if the role is analyst and uses the corresponding dictionnary
        return referenceA[grade]
    else : #else uses the senior one
        return referenceS[grade]
        
#c)
print("\nPlease be aware that the following calculator is very sensitive to input mistakes such as typping errors.\n")
status = str(input("Calculation for a new employee(y/n) :")).lower() 

if status == "y" :  #if the calculation is for a new_employee then
    years_similar = int(input("Similar role experience in years : "))
    years_lower = int(input("Lower role experience in years : "))
    role = str(input("Intended role of the employee(analyst/senioranalyst): ")).lower()
    grade = salary_grade(years_similar, years_lower) #calulates the garde based on previous inputs

    print(f"The minimum salary of this employee should be:" + str(minimum_salary(role, grade))) #uses both function to display the minimum_salary for the new employee

elif status != "y" : #if the calculation is not explicitely for a new employee
    role = str(input("What is the employee role(analyst/senioranalyst): ")).lower()
    grade = str(input("What is the employee seniority grade(A/B/C/D/E): ")).capitalize() #directly asks for the grade as per the instructions, not to roam over the years
    current_salary = int(input("What is the employee current salary: "))

    if minimum_salary(role, grade) < current_salary : #checks if the actual salary is below or above the reference tab provided in the instructions
        print(f"The salary is above average considereing : senoriety and role withon the company")
    elif current_salary < minimum_salary(role, grade):
        print(f"The salary is bellow average considering : senoriety and role within the company")