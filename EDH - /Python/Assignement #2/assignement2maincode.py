import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

df = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Python/Data/global_climate_events_economic_impact_2020_2025.csv")

oneone = df.groupby("year")["economic_impact_million_usd"].sum()
fig, axes = plt.subplots(figsize=(10, 6))
oneone.plot(kind="bar", ax=axes)
plt.title("Total Economic Impact by Year")
plt.xlabel("Year")
plt.ylabel("Economic Impact (Million USD)")
plt.xticks(rotation=45)

onetwo = df.groupby("country")["economic_impact_million_usd"].sum()
fig, axes = plt.subplots(figsize=(15, 8))
onetwo.plot(kind="bar", ax=axes)
plt.title("Total Economic Impact by Country")
plt.xlabel("Country")
plt.ylabel("Economic Impact (Million USD)")
plt.xticks(rotation=45)

plt.figure(figsize=(10, 6))
plt.scatter(df["severity"], df["economic_impact_million_usd"])
plt.xlabel("Severity Score")
plt.ylabel("Economic Impact (Million USD)")
plt.title("Economic Impact vs Severity Score")
plt.show()

def estimate_regression(df, dependent_variable, independent_variables, include_intercept=True):
    if dependent_variable not in df.columns:
        return print("Error the column names isn't contained in the df") #validate the dependent variable
    
    if type(independent_variables) == str:
        variables_to_check = [independent_variables]
        vars_formula = independent_variables #handle variable by type
    else:
        variables_to_check = independent_variables #validates independent var
        vars_formula = ""
        for i, var in enumerate(independent_variables):
            if i == 0:
                vars_formula = var
            else:
                vars_formula = vars_formula + " + " + var 
    
    for var in variables_to_check:
        if var not in df.columns:
            return None 
    
    if include_intercept: #creates the regression formula 
        formula = f"{dependent_variable} ~ {vars_formula}"
    else:
        formula = f"{dependent_variable} ~ {vars_formula} - 1"
    
    model = smf.ols(formula, data=df)
    return model.fit() #fit the model

def get_regression_parameters(regression_results, independent_variables, intercept_included):
    parameters = {} #create the dictionnary to store the values
    
    if intercept_included:
        parameters["intercept"] = regression_results.params["Intercept"] #adds intercept
    
    if type(independent_variables) == str:
        parameters[independent_variables] = regression_results.params[independent_variables] #handle variable by type
    else:
        for var in independent_variables:
            parameters[var] = regression_results.params[var] #extract coefficients
    

    return parameters

def get_significance_parameters(regression_results, independent_variables, intercept_included):
    significance= {}    #initialise dictionnary
    
    if intercept_included:
        significance["intercept"] = regression_results.pvalues["Intercept"] < 0.05 #adds intercept
    
    if type(independent_variables) == str:
        significance[independent_variables] =regression_results.pvalues[independent_variables] < 0.05 #handle variable by type
    else:
        for var in independent_variables:
            significance[var] = regression_results.pvalues[var] < 0.05 #significance test
    return significance

def format_regression_results(regression_results, dependent_variable, independent_variables, intercept_included):
    params = get_regression_parameters(regression_results, independent_variables, intercept_included)    
    significance= get_significance_parameters(regression_results, independent_variables, intercept_included) #get the data
    
    equation = f"{dependent_variable} = "  #start creating the equation
    

    if intercept_included:
        intercept_val = params["intercept"]   
        intercept_star = "*" if significance["intercept"]else ""
        equation += f"{intercept_val:.4f} intercept{intercept_star}" #add the intercept. 
    
    if type(independent_variables) == str: #handling variables
        var = independent_variables
        coef = params[var]  
        star = "*"if significance[var] else "" #markers after significant variables
        if intercept_included:    
            equation += f" + {coef:.4f} {var}{star}" #rounds results
        else:
            equation += f"{coef:.4f}{var}{star}"
    else:
        for i, var in enumerate(independent_variables):
            coef = params[var]
            star = "*" if significance[var] else ""
            if intercept_included or i > 0:
                equation += f" + {coef:.4f} {var}{star}"
            else:
                equation += f"{coef:.4f} {var}{star}"
    
    return equation

#Regressions Q3 1/2:
independent_vars_1 = ["severity", "duration_days", "affected_population",  "injuries"]
results1 = estimate_regression(df, "economic_impact_million_usd",independent_vars_1, True)
if results1:
    print(format_regression_results(results1, "economic_impact_million_usd", independent_vars_1, True))
    print(f"R-squared: {results1.rsquared}")

independent_vars_2 = independent_vars_1 + ["total_casualties", "response_time_hours"] 
results2 = estimate_regression(df, "economic_impact_million_usd", independent_vars_2,True)  
if results2:
    print(format_regression_results(results2,"economic_impact_million_usd", independent_vars_2, True))
    print(f"R-squared: {results2.rsquared}")