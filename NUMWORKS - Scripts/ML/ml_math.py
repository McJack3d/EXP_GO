import math

def menu():
    print("\n--- ML MATH & STATS ---")
    print("1. Regression Metrics (MSE, R2)")
    print("2. Distances (Eucl, Manh)")
    print("3. Activation (Sigmoid, Softmax)")
    print("4. Basic Stats (Mean, Var)")
    print("0. Menu Principal")
    try:
        return int(input("Choix: "))
    except:
        return -1

# --- 1. REGRESSION METRICS ---
def reg_metrics():
    print("\n--- REGRESSION METRICS ---")
    try:
        y_true_str = input("Y Reels: ")
        y_pred_str = input("Y Predits: ")
        Y = [float(x) for x in y_true_str.split(',')]
        P = [float(x) for x in y_pred_str.split(',')]
        n = len(Y)
        
        if n != len(P): return
        
        # Errors
        errs = [Y[i] - P[i] for i in range(n)]
        abs_errs = [abs(e) for e in errs]
        sq_errs = [e**2 for e in errs]
        
        mae = sum(abs_errs) / n
        mse = sum(sq_errs) / n
        rmse = math.sqrt(mse)
        
        # R2 Score
        y_mean = sum(Y) / n
        ss_tot = sum([(y - y_mean)**2 for y in Y])
        ss_res = sum(sq_errs)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        print("-" * 20)
        print(f"MAE  (Mean Abs Err): {mae:.4f}")
        print(f"MSE  (Mean Sq Err):  {mse:.4f}")
        print(f"RMSE (Root MSE):     {rmse:.4f}")
        print(f"R2 Score:            {r2:.4f}")
        print("-" * 20)
        print("R2 proche 1 = Bon modele")
        print("R2 proche 0 = Modele moyen (baseline)")
    except:
        print("Erreur")

# --- 2. DISTANCES ---
def dist():
    print("\n--- DISTANCES ---")
    try:
        p1_str = input("Pt 1 (x,y,..): ")
        p2_str = input("Pt 2 (x,y,..): ")
        P1 = [float(x) for x in p1_str.split(',')]
        P2 = [float(x) for x in p2_str.split(',')]
        
        if len(P1) != len(P2): return
        
        # Euclidean (L2)
        sum_sq = sum([(P1[i]-P2[i])**2 for i in range(len(P1))])
        eucl = math.sqrt(sum_sq)
        
        # Manhattan (L1)
        manh = sum([abs(P1[i]-P2[i]) for i in range(len(P1))])
        
        print(f"Euclidienne (L2): {eucl:.4f}")
        print(f"Manhattan (L1):   {manh:.4f}")
    except:
        print("Erreur")

# --- 3. ACTIVATION FUNCTIONS ---
def activation():
    print("\n--- ACTIVATION ---")
    print("1. Sigmoid (Logistic)")
    print("2. Softmax (Multi-class)")
    c = input("Choix: ")
    
    if c == '1':
        z = float(input("z: "))
        sig = 1 / (1 + math.exp(-z))
        print(f"Sigmoid(z) = {sig:.4f}")
        print("Prediction: " + ("1" if sig>0.5 else "0"))
        
    elif c == '2':
        z_str = input("Logits (z1,z2..): ")
        Z = [float(x) for x in z_str.split(',')]
        # Softmax formula
        exp_z = [math.exp(z) for z in Z]
        sum_exp = sum(exp_z)
        probs = [e/sum_exp for e in exp_z]
        
        print("Probabilites:")
        for i, p in enumerate(probs):
            print(f"Class {i}: {p:.4f}")

# --- 4. BASIC STATS ---
def stats():
    try:
        data_str = input("Data: ")
        D = [float(x) for x in data_str.split(',')]
        n = len(D)
        if n == 0: return
        
        mean = sum(D) / n
        var = sum([(d-mean)**2 for d in D]) / n # Population var
        std = math.sqrt(var)
        
        D.sort()
        med = D[n//2] if n%2==1 else (D[n//2-1]+D[n//2])/2
        
        print(f"Mean: {mean:.4f}")
        print(f"Median: {med:.4f}")
        print(f"Std Dev: {std:.4f}")
        print(f"Min/Max: {D[0]}/{D[-1]}")
    except:
        print("Erreur")

while True:
    c = menu()
    if c == 0: break
    elif c == 1: reg_metrics()
    elif c == 2: dist()
    elif c == 3: activation()
    elif c == 4: stats()