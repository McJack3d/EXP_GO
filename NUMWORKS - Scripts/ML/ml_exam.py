import math

def menu():
    print("\n--- ML ALGORITHMS ---")
    print("1. Best Tree Split (Gini)")
    print("2. Linear Reg (Simple)")
    print("3. Gradient Descent (Simu)")
    print("4. Lasso/Ridge Cost")
    print("0. Menu Principal")
    try:
        return int(input("Choix: "))
    except:
        return -1

# --- 1. FIND BEST SPLIT (Pour Exam Q2) ---
def best_split():
    print("\n--- BEST SPLIT FINDER ---")
    print("Trouve le meilleur seuil de coupure.")
    print("Format: valeurs X et classes Y (0/1)")
    print("Ex: X=[4,6,7], Y=[0,1,0]")
    
    try:
        x_str = input("X (valeurs): ")
        y_str = input("Y (classes): ")
        X = [float(v) for v in x_str.split(',')]
        Y = [int(v) for v in y_str.split(',')]
        
        if len(X) != len(Y):
            print("Err: X et Y de longueurs differentes")
            return

        # Combine and sort
        data = sorted(zip(X, Y))
        X_s = [d[0] for d in data]
        Y_s = [d[1] for d in data]
        
        n = len(Y_s)
        best_gini = 1.0
        best_split = None
        
        print("\nTest des seuils...")
        # Test splits between points
        for i in range(n-1):
            if X_s[i] == X_s[i+1]: continue # Skip duplicate values
            
            split_val = (X_s[i] + X_s[i+1]) / 2
            
            # Left group (<= split)
            left_y = Y_s[:i+1]
            n_left = len(left_y)
            p1_left = sum(left_y) / n_left
            gini_left = 1 - (p1_left**2 + (1-p1_left)**2)
            
            # Right group (> split)
            right_y = Y_s[i+1:]
            n_right = len(right_y)
            p1_right = sum(right_y) / n_right
            gini_right = 1 - (p1_right**2 + (1-p1_right)**2)
            
            # Weighted Gini
            w_gini = (n_left/n)*gini_left + (n_right/n)*gini_right
            
            print(f"Split {split_val:.2f} -> Gini {w_gini:.4f}")
            
            if w_gini < best_gini:
                best_gini = w_gini
                best_split = split_val
                
        print("-" * 20)
        print(f"BEST SPLIT: {best_split}")
        print(f"MIN IMPURITY: {best_gini:.4f}")
        
    except:
        print("Erreur saisie")

# --- 2. SIMPLE LINEAR REGRESSION ---
def lin_reg():
    print("\n--- SIMPLE LIN REG ---")
    print("y = mx + b")
    try:
        x_str = input("X: ")
        y_str = input("Y: ")
        X = [float(v) for v in x_str.split(',')]
        Y = [float(v) for v in y_str.split(',')]
        
        n = len(X)
        sum_x = sum(X)
        sum_y = sum(Y)
        sum_xy = sum([X[i]*Y[i] for i in range(n)])
        sum_xx = sum([X[i]**2 for i in range(n)])
        
        m = (n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x**2)
        b = (sum_y - m*sum_x) / n
        
        print(f"m (pente) = {m:.4f}")
        print(f"b (intercept) = {b:.4f}")
        print(f"Eq: y = {m:.2f}x + {b:.2f}")
    except:
        print("Erreur")

# --- 3. GRADIENT DESCENT SIMU ---
def grad_descent():
    print("\n--- GRADIENT DESCENT ---")
    print("Simule 1 pas de mise a jour (Batch)")
    print("Hypothese: h(x) = t0 + t1*x")
    try:
        t0 = float(input("Theta0 (Intercept) actuel: "))
        t1 = float(input("Theta1 (Pente) actuel: "))
        lr = float(input("Learning Rate (alpha): "))
        
        x_str = input("X data: ")
        y_str = input("Y data: ")
        X = [float(v) for v in x_str.split(',')]
        Y = [float(v) for v in y_str.split(',')]
        m = len(X)
        
        # Gradient dJ/dt0 = (1/m) * sum(h(x) - y)
        # Gradient dJ/dt1 = (1/m) * sum((h(x) - y)*x)
        
        sum_err = 0
        sum_err_x = 0
        
        print("\nDetails erreurs:")
        for i in range(m):
            pred = t0 + t1*X[i]
            err = pred - Y[i]
            sum_err += err
            sum_err_x += err * X[i]
            print(f"Pt {i}: Pred={pred:.2f}, Err={err:.2f}")
            
        grad0 = (1/m) * sum_err
        grad1 = (1/m) * sum_err_x
        
        new_t0 = t0 - lr * grad0
        new_t1 = t1 - lr * grad1
        
        print("-" * 20)
        print(f"Grad0: {grad0:.4f}")
        print(f"Grad1: {grad1:.4f}")
        print(f"NEW Theta0: {new_t0:.4f}")
        print(f"NEW Theta1: {new_t1:.4f}")
        
    except:
        print("Erreur")

# --- 4. LASSO/RIDGE COST ---
def reg_cost():
    print("\n--- COST FUNCTION ---")
    print("Calc Cout Total = MSE + Penalite")
    try:
        mse = float(input("MSE (Loss): "))
        lam = float(input("Lambda: "))
        coefs_str = input("Coefs (beta) sep ,: ")
        coefs = [float(x) for x in coefs_str.split(',')]
        
        l1_norm = sum([abs(c) for c in coefs])
        l2_norm_sq = sum([c**2 for c in coefs])
        
        cost_lasso = mse + lam * l1_norm
        cost_ridge = mse + lam * l2_norm_sq
        
        print("-" * 20)
        print(f"L1 (Lasso) Penalty: {lam*l1_norm:.4f}")
        print(f"TOTAL LASSO COST: {cost_lasso:.4f}")
        print("-" * 20)
        print(f"L2 (Ridge) Penalty: {lam*l2_norm_sq:.4f}")
        print(f"TOTAL RIDGE COST: {cost_ridge:.4f}")
        print("-" * 20)
        print("Rappel: Lasso met coefs a 0 (Selection)")
        print("Rappel: Ridge reduit coefs (Shrinkage)")
        
    except:
        print("Erreur")

while True:
    c = menu()
    if c == 0: break
    elif c == 1: best_split()
    elif c == 2: lin_reg()
    elif c == 3: grad_descent()
    elif c == 4: reg_cost()