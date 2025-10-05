import pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Load datasets
try:
    df_train, meta_train = pyreadstat.read_sav('EXP_GO/Project_Dataset_J.sav', apply_value_formats=True)
    print("Training dataset loaded!")
except:
    df_train, meta_train = pyreadstat.read_sav('/Users/alexandrebredillot/Documents/GitHub/EXP_GO/Project_Dataset_J.sav', apply_value_formats=True)
    print("Training dataset loaded!")

try:
    df_base, meta_base = pyreadstat.read_sav('EXP_GO/Project_Dataset_Base.sav', apply_value_formats=True)
    print("Base dataset loaded!")
except:
    df_base, meta_base = pyreadstat.read_sav('/Users/alexandrebredillot/Documents/GitHub/EXP_GO/Project_Dataset_Base.sav', apply_value_formats=True)
    print("Base dataset loaded!")

print(f"Base dataset shape: {df_base.shape}")
print(f"Base columns: {list(df_base.columns)}")
print(f"Train dataset shape: {df_train.shape}")
print(f"Train columns: {list(df_train.columns)}")

# ============================================================================
# 1. RESPONSE RATE BY MARITAL STATUS
# ============================================================================

def analyze_response_by_marital_status(df):
    """Analyze response rate by marital status"""
    print("\n" + "="*50)
    print("RESPONSE RATE BY MARITAL STATUS")
    print("="*50)
    
    # Find response and marital status columns
    response_col = None
    marital_col = None
    
    for col in df.columns:
        if 'response' in col.lower():
            response_col = col
        if 'marital' in col.lower():
            marital_col = col
    
    if response_col is None or marital_col is None:
        print("Required columns not found. Skipping analysis.")
        return None
    
    print(f"Using columns: Response={response_col}, Marital={marital_col}")
    
    # Calculate response rates
    marital_response = df.groupby(marital_col).agg({
        response_col: ['count', 'sum', 'mean']
    }).round(3)
    
    marital_response.columns = ['Total_Customers', 'Responders', 'Response_Rate']
    marital_response['Response_Rate_Pct'] = (marital_response['Response_Rate'] * 100).round(1)
    
    print("\nResponse Rate by Marital Status:")
    print(marital_response)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    marital_response['Response_Rate_Pct'].plot(kind='bar', ax=ax1, color='skyblue')
    ax1.set_title('Response Rate by Marital Status')
    ax1.set_ylabel('Response Rate (%)')
    ax1.set_xlabel('Marital Status')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    for i, v in enumerate(marital_response['Response_Rate_Pct']):
        ax1.text(i, v + 0.5, f'{v}%', ha='center', va='bottom')
    
    # Pie chart
    marital_response['Total_Customers'].plot(kind='pie', ax=ax2, autopct='%1.1f%%')
    ax2.set_title('Customer Distribution by Marital Status')
    ax2.set_ylabel('')
    
    plt.tight_layout()
    plt.show()
    
    # Statistical test
    contingency_table = pd.crosstab(df[marital_col], df[response_col])
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    print(f"\nChi-square test: {chi2:.3f}, p-value: {p_value:.3f}")
    if p_value < 0.05:
        print("✓ Statistically significant relationship between marital status and response rate")
    else:
        print("✗ No statistically significant relationship found")
    
    return marital_response

# ============================================================================
# 2. CORRELATION HEATMAP
# ============================================================================

def create_correlation_heatmap(df):
    """Create correlation heatmap between numerical variables"""
    print("\n" + "="*50)
    print("CORRELATION HEATMAP")
    print("="*50)
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove ID column if present
    if 'ID' in numerical_cols:
        numerical_cols.remove('ID')
    
    print(f"Numerical columns for correlation: {numerical_cols}")
    
    # Limit to key variables (max 15 for readability)
    if len(numerical_cols) > 15:
        numerical_cols = numerical_cols[:15]
    
    if len(numerical_cols) < 2:
        print("Not enough numerical variables for correlation analysis")
        return None
    
    # Calculate correlation matrix
    correlation_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, 
                mask=mask,
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                linewidths=0.5,
                fmt='.2f')
    
    plt.title('Correlation Heatmap - Numerical Variables', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return correlation_matrix

# ============================================================================
# 3. PREPARE DATA AND MODELS
# ============================================================================

def prepare_and_train_models(df):
    """Prepare data and train all models"""
    print("\n" + "="*50)
    print("PREPARING MODELS ON TRAINING DATASET")
    print("="*50)
    
    # Find target columns
    wine_col = None
    gold_col = None
    for col in df.columns:
        if 'wine' in col.lower():
            wine_col = col
        elif 'gold' in col.lower():
            gold_col = col
    
    print(f"Found target columns: Wine={wine_col}, Gold={gold_col}")
    
    if wine_col is None or gold_col is None:
        raise ValueError("Target columns not found in training dataset")
    
    # Prepare features
    feature_cols = [c for c in df.columns if c not in [wine_col, gold_col]]
    X = df[feature_cols]
    y_wine = df[wine_col].astype(int)
    y_gold = df[gold_col].astype(int)
    
    print(f"Features: {len(feature_cols)} columns")
    print(f"Wine buyer rate: {y_wine.mean():.3f}")
    print(f"Gold buyer rate: {y_gold.mean():.3f}")
    
    # Split data
    X_train, X_test, y_wine_train, y_wine_test = train_test_split(
        X, y_wine, test_size=0.3, random_state=42, stratify=y_wine
    )
    y_gold_train = y_gold.loc[y_wine_train.index]
    y_gold_test = y_gold.loc[y_wine_test.index]
    
    # Preprocessing
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numerical_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", __import__("sklearn.preprocessing").preprocessing.OneHotEncoder(handle_unknown="ignore"))
            ]), categorical_cols)
        ]
    )
    
    # Train models
    models = {}
    
    print("Training Logistic Regression models...")
    # Logistic Regression
    wine_lr = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    gold_lr = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000, random_state=42))])
    wine_lr.fit(X_train, y_wine_train)
    gold_lr.fit(X_train, y_gold_train)
    models['Logistic Regression'] = {'wine': wine_lr, 'gold': gold_lr}
    
    print("Training Neural Network models...")
    # Neural Network
    wine_nn = Pipeline([("prep", preprocessor), ("clf", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))])
    gold_nn = Pipeline([("prep", preprocessor), ("clf", MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42))])
    wine_nn.fit(X_train, y_wine_train)
    gold_nn.fit(X_train, y_gold_train)
    models['Neural Network'] = {'wine': wine_nn, 'gold': gold_nn}
    
    print("Training Decision Tree models...")
    # Decision Tree
    wine_dt = Pipeline([("prep", preprocessor), ("clf", DecisionTreeClassifier(random_state=42, max_depth=10))])
    gold_dt = Pipeline([("prep", preprocessor), ("clf", DecisionTreeClassifier(random_state=42, max_depth=10))])
    wine_dt.fit(X_train, y_wine_train)
    gold_dt.fit(X_train, y_gold_train)
    models['Decision Tree'] = {'wine': wine_dt, 'gold': gold_dt}
    
    print("All models trained successfully!")
    
    return models, X_train, X_test, y_wine_train, y_wine_test, y_gold_train, y_gold_test

# ============================================================================
# 4. ROC CURVES
# ============================================================================

def plot_roc_curves(models, X_test, y_wine_test, y_gold_test):
    """Plot ROC curves for all models"""
    print("\n" + "="*50)
    print("PLOTTING ROC CURVES")
    print("="*50)
    
    # Wine ROC Curves
    plt.figure(figsize=(10, 8))
    for model_name, model_dict in models.items():
        scores = model_dict['wine'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_wine_test, scores)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Wine Buyers')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Gold ROC Curves
    plt.figure(figsize=(10, 8))
    for model_name, model_dict in models.items():
        scores = model_dict['gold'].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_gold_test, scores)
        auc_score = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - Gold Buyers')
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================================================================
# 5. LIFT CHARTS (FIXED)
# ============================================================================

def plot_lift_charts(models, X_test, y_wine_test, y_gold_test):
    """Plot lift charts for all models"""
    print("\n" + "="*50)
    print("PLOTTING LIFT CHARTS")
    print("="*50)
    
    # Wine Lift Chart
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for model_name, model_dict in models.items():
        scores = model_dict['wine'].predict_proba(X_test)[:, 1]
        df_lift = pd.DataFrame({'score': scores, 'actual': y_wine_test})
        df_lift = df_lift.sort_values('score', ascending=False)
        
        # Fix the duplicate edges issue by using duplicates='drop'
        try:
            df_lift['decile'] = pd.qcut(df_lift['score'], 10, labels=False, duplicates='drop') + 1
        except ValueError:
            # If still fails, use rank-based approach
            df_lift['decile'] = pd.qcut(df_lift['score'].rank(method='first'), 10, labels=False) + 1
        
        lift_data = df_lift.groupby('decile')['actual'].mean() / y_wine_test.mean()
        plt.plot(range(1, len(lift_data) + 1), lift_data, marker='o', label=model_name)
    
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Decile')
    plt.ylabel('Lift')
    plt.title('Lift Chart - Wine Buyers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gold Lift Chart
    plt.subplot(1, 2, 2)
    for model_name, model_dict in models.items():
        scores = model_dict['gold'].predict_proba(X_test)[:, 1]
        df_lift = pd.DataFrame({'score': scores, 'actual': y_gold_test})
        df_lift = df_lift.sort_values('score', ascending=False)
        
        # Fix the duplicate edges issue
        try:
            df_lift['decile'] = pd.qcut(df_lift['score'], 10, labels=False, duplicates='drop') + 1
        except ValueError:
            df_lift['decile'] = pd.qcut(df_lift['score'].rank(method='first'), 10, labels=False) + 1
        
        lift_data = df_lift.groupby('decile')['actual'].mean() / y_gold_test.mean()
        plt.plot(range(1, len(lift_data) + 1), lift_data, marker='o', label=model_name)
    
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Decile')
    plt.ylabel('Lift')
    plt.title('Lift Chart - Gold Buyers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 6. CONFUSION MATRICES
# ============================================================================

def plot_confusion_matrices(models, X_test, y_wine_test, y_gold_test):
    """Plot confusion matrices for all models"""
    print("\n" + "="*50)
    print("PLOTTING CONFUSION MATRICES")
    print("="*50)
    
    # Wine Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Confusion Matrices - Wine Buyers', fontsize=16)
    
    for i, (model_name, model_dict) in enumerate(models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model_dict['wine'], X_test, y_wine_test, ax=axes[i], cmap='Blues'
        )
        axes[i].set_title(model_name)
    
    plt.tight_layout()
    plt.show()
    
    # Gold Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Confusion Matrices - Gold Buyers', fontsize=16)
    
    for i, (model_name, model_dict) in enumerate(models.items()):
        ConfusionMatrixDisplay.from_estimator(
            model_dict['gold'], X_test, y_gold_test, ax=axes[i], cmap='Oranges'
        )
        axes[i].set_title(model_name)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# 7. COMBINED MODEL AND DEPLOYMENT
# ============================================================================

def create_combined_model_analysis(models, X_test, y_wine_test, y_gold_test):
    """Create combined model analysis and deployment visualization"""
    print("\n" + "="*50)
    print("COMBINED MODEL ANALYSIS")
    print("="*50)
    
    # Use best model (Logistic Regression) for combined scoring
    wine_scores = models['Logistic Regression']['wine'].predict_proba(X_test)[:, 1]
    gold_scores = models['Logistic Regression']['gold'].predict_proba(X_test)[:, 1]
    
    # Standardize and combine scores
    wine_std = (wine_scores - np.mean(wine_scores)) / np.std(wine_scores)
    gold_std = (gold_scores - np.mean(gold_scores)) / np.std(gold_scores)
    combined_scores = (wine_std + gold_std) / 2
    
    # Plot distribution of combined scores
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(combined_scores, bins=30, alpha=0.7, color='purple')
    plt.axvline(np.percentile(combined_scores, 80), color='red', linestyle='--', 
                label='Top 20% Threshold')
    plt.xlabel('Combined Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Combined Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Combined model lift chart (Top 20% analysis)
    plt.subplot(1, 2, 2)
    df_combined = pd.DataFrame({
        'combined_score': combined_scores,
        'wine_actual': y_wine_test,
        'gold_actual': y_gold_test
    })
    df_combined = df_combined.sort_values('combined_score', ascending=False)
    
    # Fix decile calculation
    try:
        df_combined['decile'] = pd.qcut(df_combined['combined_score'], 10, labels=False, duplicates='drop') + 1
    except ValueError:
        df_combined['decile'] = pd.qcut(df_combined['combined_score'].rank(method='first'), 10, labels=False) + 1
    
    wine_lift = df_combined.groupby('decile')['wine_actual'].mean() / y_wine_test.mean()
    gold_lift = df_combined.groupby('decile')['gold_actual'].mean() / y_gold_test.mean()
    
    plt.plot(range(1, len(wine_lift) + 1), wine_lift, marker='o', label='Wine Lift', color='red')
    plt.plot(range(1, len(gold_lift) + 1), gold_lift, marker='s', label='Gold Lift', color='gold')
    plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Decile')
    plt.ylabel('Lift')
    plt.title('Combined Model Lift Chart')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Top 20% performance analysis
    top_20_threshold = np.percentile(combined_scores, 80)
    top_20_mask = combined_scores >= top_20_threshold
    
    print(f"\nTOP 20% PERFORMANCE ANALYSIS:")
    print(f"Wine buyer rate in top 20%: {y_wine_test[top_20_mask].mean():.3f}")
    print(f"Gold buyer rate in top 20%: {y_gold_test[top_20_mask].mean():.3f}")
    print(f"Overall wine buyer rate: {y_wine_test.mean():.3f}")
    print(f"Overall gold buyer rate: {y_gold_test.mean():.3f}")
    print(f"Wine lift: {y_wine_test[top_20_mask].mean()/y_wine_test.mean():.2f}x")
    print(f"Gold lift: {y_gold_test[top_20_mask].mean()/y_gold_test.mean():.2f}x")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    print("CUSTOMER PERSONALITY ANALYSIS - VISUALIZATION FOCUS")
    print("="*60)
    
    # 1. Response Rate by Marital Status (on BASE dataset)
    marital_analysis = analyze_response_by_marital_status(df_base)
    
    # 2. Correlation Heatmap (on BASE dataset)
    correlation_matrix = create_correlation_heatmap(df_base)
    
    # 3. Prepare models (on TRAINING dataset)
    models, X_train, X_test, y_wine_train, y_wine_test, y_gold_train, y_gold_test = prepare_and_train_models(df_train)
    
    # 4. ROC Curves
    plot_roc_curves(models, X_test, y_wine_test, y_gold_test)
    
    # 5. Lift Charts
    plot_lift_charts(models, X_test, y_wine_test, y_gold_test)
    
    # 6. Confusion Matrices
    plot_confusion_matrices(models, X_test, y_wine_test, y_gold_test)
    
    # 7. Combined Model Analysis
    create_combined_model_analysis(models, X_test, y_wine_test, y_gold_test)
    
    print("\n" + "="*60)
    print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()