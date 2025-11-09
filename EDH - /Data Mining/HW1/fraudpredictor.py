import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

df = pd.read_csv("/Users/alexandrebredillot/Documents/GitHub/EXP_GO/EDH - /Data Mining/HW1/fraud_homework_features.csv")

# The goal of this script is to understand data and build predictions accordingly. From a record of 10000 values representing transactions, we must flag using bibary encoding wether a transaction is fraud or not using 4 different features.
"""
print(df.describe())

plt.figure()
plt.scatter(df["Transaction_Amount"], df.index)
plt.title("Transaction Amount")
plt.xlabel("Transaction Amount")
plt.ylabel("Index")
plt.show()

plt.figure()
plt.scatter(df["Items_in_Cart"], df.index)
plt.title("Items in Cart")
plt.xlabel("Items in Cart")
plt.ylabel("Index")
plt.show()

plt.figure()
plt.scatter(df["IP_Reputation_Score"], df.index)
plt.title("IP Reputation Score")
plt.xlabel("IP Reputation Score")
plt.ylabel("Index")
plt.show()

plt.figure()
plt.scatter(df["Transaction_vs_Avg"], df.index)
plt.title("Transaction vs Avg")
plt.xlabel("Transaction vs Avg")
plt.ylabel("Index")
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(df["Transaction_Amount"])
plt.title("Boxplot - Transaction Amount")
plt.ylabel("Transaction Amount")
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(df["Items_in_Cart"])
plt.title("Boxplot - Items in Cart")
plt.ylabel("Items in Cart")
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(df["IP_Reputation_Score"])
plt.title("Boxplot - IP Reputation Score")
plt.ylabel("IP Reputation Score")
plt.show()

plt.figure(figsize=(10, 6))
plt.boxplot(df["Transaction_vs_Avg"])
plt.title("Boxplot - Transaction vs Avg")
plt.ylabel("Transaction vs Avg")
plt.show()

# After constructing the plots, we notice that the Transaction_Amount variable presents some very aberrant values (distant from the rest), a fairly high standard deviation, thus a strong dispersion of values. This is also the case for the Transaction_vs_Avg variable which also shows very high dispersion.
# Since there is a high presence of extreme values and that the values aren't normally distributed at all(very assymetric distributions), opting for an isolation forest strategy sounds pertinent.
"""
#mamamama
X_scaled = StandardScaler().fit_transform(df)

# Print first 5 rows (numpy array)
print(X_scaled[:5])

# Or convert back to DataFrame to use .head()
X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)
print(X_scaled_df.head())

# Fix: X_scaled is numpy array, not DataFrame
print(X_scaled[:5])  # Print first 5 rows

# Step 1: Train Isolation Forest model
# contamination = expected proportion of outliers (frauds) in your dataset
# Adjust this based on your data (typically 0.01 to 0.1 for fraud detection)
iso_forest = IsolationForest(
    contamination=0.05,  # Assume 5% of transactions are fraudulent
    random_state=42,
    n_estimators=100
)

# Step 2: Fit the model and predict
predictions = iso_forest.fit_predict(X_scaled)

# Step 3: Convert predictions to binary (1 = normal, -1 = fraud)
# Convert -1 (fraud) to 1, and 1 (normal) to 0 for easier interpretation
df['Is_Fraud'] = (predictions == -1).astype(int)

# Step 4: Get anomaly scores (lower = more anomalous/fraudulent)
df['Anomaly_Score'] = iso_forest.decision_function(X_scaled)

# Step 5: Analyze results
print("\n=== Fraud Detection Results ===")
print(f"Total transactions: {len(df)}")
print(f"Flagged as fraud: {df['Is_Fraud'].sum()}")
print(f"Fraud percentage: {(df['Is_Fraud'].sum() / len(df)) * 100:.2f}%")

print("\n=== Sample of Flagged Fraudulent Transactions ===")
print(df[df['Is_Fraud'] == 1].head(10))

# Step 6: Visualize results
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Transaction Amount vs Anomaly Score
axes[0, 0].scatter(df['Transaction_Amount'], df['Anomaly_Score'], 
                   c=df['Is_Fraud'], cmap='coolwarm', alpha=0.6)
axes[0, 0].set_title('Transaction Amount vs Anomaly Score')
axes[0, 0].set_xlabel('Transaction Amount')
axes[0, 0].set_ylabel('Anomaly Score')

# Plot 2: Items in Cart vs Anomaly Score
axes[0, 1].scatter(df['Items_in_Cart'], df['Anomaly_Score'], 
                   c=df['Is_Fraud'], cmap='coolwarm', alpha=0.6)
axes[0, 1].set_title('Items in Cart vs Anomaly Score')
axes[0, 1].set_xlabel('Items in Cart')
axes[0, 1].set_ylabel('Anomaly Score')

# Plot 3: IP Reputation Score vs Anomaly Score
axes[1, 0].scatter(df['IP_Reputation_Score'], df['Anomaly_Score'], 
                   c=df['Is_Fraud'], cmap='coolwarm', alpha=0.6)
axes[1, 0].set_title('IP Reputation Score vs Anomaly Score')
axes[1, 0].set_xlabel('IP Reputation Score')
axes[1, 0].set_ylabel('Anomaly Score')

# Plot 4: Transaction vs Avg vs Anomaly Score
axes[1, 1].scatter(df['Transaction_vs_Avg'], df['Anomaly_Score'], 
                   c=df['Is_Fraud'], cmap='coolwarm', alpha=0.6)
axes[1, 1].set_title('Transaction vs Avg vs Anomaly Score')
axes[1, 1].set_xlabel('Transaction vs Avg')
axes[1, 1].set_ylabel('Anomaly Score')

plt.tight_layout()
plt.show()

# Step 7: Export results to CSV
df.to_csv('fraud_predictions.csv', index=False)
print("\n✅ Predictions exported to 'fraud_predictions.csv'")

# Step 8: Feature importance analysis
print("\n=== Feature Statistics for Fraudulent Transactions ===")
fraud_df = df[df['Is_Fraud'] == 1]
normal_df = df[df['Is_Fraud'] == 0]

for col in ['Transaction_Amount', 'Items_in_Cart', 'IP_Reputation_Score', 'Transaction_vs_Avg']:
    print(f"\n{col}:")
    print(f"  Fraud mean: {fraud_df[col].mean():.2f}")
    print(f"  Normal mean: {normal_df[col].mean():.2f}")
    print(f"  Fraud std: {fraud_df[col].std():.2f}")
    print(f"  Normal std: {normal_df[col].std():.2f}")