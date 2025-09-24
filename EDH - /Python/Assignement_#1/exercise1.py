#a)
years_similar = 0
years_lower = 0
years_total = 0
grade = ""

def salary_grade(years_similar, years_lower):
    years_total = years_lower * 0.5 + years_similar
    if years_total < 2 :
        grade = "A"
    elif 2 <= years_total <= 5 :
        grade = "B"
    elif 6 <= years_total <= 11 :
        grade = "C"
    elif 12 <= years_total <= 19 :
        grade = "D"
    elif years_total >= 20 :
        grade = "E"
    return grade

#b)
role = ""
referenceA= { "A" : 2700, "B" : 2900, "C" : 3300, "D" : 3600, "E" : 3900}
referenceS= { "A" : 3800, "B" : 4200, "C" : 4400, "D" : 4700, "E" : 5100}

def minimum_salary(role, grade):

    if role == "analyst" :
        return referenceA[grade]
    else :
        return referenceS[grade]
        
#c)
status = str(input("Calculation for a new employee(y/n) :")).lower()

if status == "y" :
    years_similar = int(input("Similar role experience in years : "))
    years_lower = int(input("Lower role experience in years : "))
    role = str(input("Intended role of the employee(analyst/senioranalyst): ")).lower()
    grade = salary_grade(years_similar, years_lower)

    print(f"The minimum salary of this employee should be:" + str(minimum_salary(role, grade)))

elif status != "y" :
    role = str(input("What is the employee role(analyst/senioranalyst): ")).lower()
    grade = str(input("What is the employee seniority grade(A/B/C/D/E): ")).capitalize()
    current_salary = int(input("What is the employee current salary: "))

    if minimum_salary(role, grade) > current_salary :
        high_salary = (minimum_salary(role, grade) - current_salary)
        print("{(minimum_salary(role, grade) - current_salary)}should)
    elif current_salary > minimum_salary(role, grade):
        print(str(minimum_salary(role, grade) - current_salary))