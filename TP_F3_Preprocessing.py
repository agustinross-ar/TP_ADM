import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Ensure the output directory exists
os.makedirs("output", exist_ok=True)

# -------------------------------------------------------------
# Base Data Loading and Phase 2 Cleaning
# -------------------------------------------------------------
csv_path = 'bodyfat.csv'
if not os.path.exists(csv_path):
    csv_path = 'TP_ADM/bodyfat.csv'
if not os.path.exists(csv_path):
    csv_path = '/home/fzacchigna/ceia/2-amq/TP_ADM/bodyfat.csv'

print(f"Loading dataset from: {csv_path}")
df = pd.read_csv(csv_path)

# Phase 2 Anomaly Cleaning:
# 1. Drop rows with BodyFat <= 0
df_clean = df[df['BodyFat'] > 0].copy()
print(f"Removed rows with BodyFat <= 0 (original size: {len(df)}, clean size: {len(df_clean)})")

# 2. Impute height anomaly (29.5 inches) with median
min_height_idx = df_clean['Height'].idxmin()
mediana_height = df_clean['Height'].median()
df_clean.loc[min_height_idx, 'Height'] = mediana_height
print(f"Imputed height anomaly at index {min_height_idx} with median height ({mediana_height} inches)")

# -------------------------------------------------------------
# Step 0: Remove Density column
# -------------------------------------------------------------
print("\n--- Step 0: Removing Density column ---")
df_prep = df_clean.drop(columns="Density")
print(f"Dataset columns after dropping Density: {list(df_prep.columns)}")

# -------------------------------------------------------------
# Step 1: 70/30 Split
# -------------------------------------------------------------
print("\n--- Step 1: Splitting data (70/30) ---")
X = df_prep.drop(columns="BodyFat")
y = df_prep["BodyFat"]

# Using random_state=42 for reproducibility
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print(f"Train set shape: Features: {X_train.shape}, Target: {y_train.shape}")
print(f"Test set shape:  Features: {X_test.shape}, Target: {y_test.shape}")

# -------------------------------------------------------------
# Step 2: Print boxplots for all columns of training data
# -------------------------------------------------------------
print("\n--- Step 2: Generating initial boxplots ---")
# Combine y_train and X_train for visualizing all columns in training set
df_train = pd.concat([y_train, X_train], axis=1)

# Plot boxplots in a grid
cols = df_train.columns
fig, axes = plt.subplots(4, 4, figsize=(16, 16))
axes = axes.flatten()

for i, col in enumerate(cols):
    sns.boxplot(y=df_train[col], ax=axes[i], color='skyblue')
    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].set_ylabel('')
    axes[i].grid(True, linestyle='--', alpha=0.5)

# Delete unused axes
for j in range(len(cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Boxplots - Training Data (Con Outliers)", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plot_path_1 = "output/01_boxplots_con_outliers.png"
plt.savefig(plot_path_1, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved initial boxplots to {plot_path_1}")

# -------------------------------------------------------------
# Step 3: Outlier removal placeholder
# -------------------------------------------------------------
print("\n--- Step 3: Outlier removal placeholder ---")
def remove_outliers(X_df, y_df):
    """
    Placeholder function for outlier removal.
    Currently returns unchanged copies of X and y.
    """
    # Placeholder comment: here we will add outlier detection and removal/imputation logic
    # after analyzing the plots generated in Step 2.
    return X_df.copy(), y_df.copy()

X_train_clean, y_train_clean = remove_outliers(X_train, y_train)
print("Outlier removal placeholder executed. Training data size remains identical.")

# -------------------------------------------------------------
# Step 4: Print boxplots without outliers (comparison)
# -------------------------------------------------------------
print("\n--- Step 4: Generating comparative boxplots ---")
fig, axes = plt.subplots(4, 4, figsize=(18, 18))
axes = axes.flatten()

# Recombine target and features for clean data
df_train_clean = pd.concat([y_train_clean, X_train_clean], axis=1)

for i, col in enumerate(cols):
    # Create a comparative dataframe for this column
    comp_df = pd.DataFrame({
        'Con Outliers': df_train[col],
        'Sin Outliers': df_train_clean[col]
    })
    sns.boxplot(data=comp_df, ax=axes[i], palette="Set2")
    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].grid(True, linestyle='--', alpha=0.5)

for j in range(len(cols), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Comparativa de Boxplots: Con vs. Sin Outliers", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plot_path_2 = "output/02_boxplots_comparacion.png"
plt.savefig(plot_path_2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved comparative boxplots to {plot_path_2}")

# -------------------------------------------------------------
# Step 5: Fit scaler (only with train data)
# -------------------------------------------------------------
print("\n--- Step 5: Fitting StandardScaler on training features ---")
scaler = StandardScaler()
# Note: we only fit the scaler on features, excluding target variable y
scaler.fit(X_train_clean)
print("StandardScaler fitted on clean training features.")

# -------------------------------------------------------------
# Step 6: Apply scaler to train data
# -------------------------------------------------------------
print("\n--- Step 6: Applying scaler to train data ---")
X_train_scaled = scaler.transform(X_train_clean)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train_clean.columns, index=X_train_clean.index)
print("Training features scaled.")

# -------------------------------------------------------------
# Step 7: Apply scaler to test data
# -------------------------------------------------------------
print("\n--- Step 7: Applying training scaler to test data ---")
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
print("Test features scaled.")

# Display scale statistics to verify correctness
print("\n--- Preprocessing Phase 3 Verification ---")
print("\nFirst 3 rows of scaled training features:")
print(X_train_scaled_df.head(3))
print("\nScaling Statistics (Train - Mean should be approx 0, Std approx 1):")
stats_train = pd.DataFrame({
    'Mean': X_train_scaled_df.mean(),
    'Std': X_train_scaled_df.std()
})
print(stats_train.round(4))

print("\nScaling Statistics (Test - Mean/Std will deviate slightly):")
stats_test = pd.DataFrame({
    'Mean': X_test_scaled_df.mean(),
    'Std': X_test_scaled_df.std()
})
print(stats_test.round(4))
