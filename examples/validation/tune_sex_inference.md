Download data and run Gnomon to generate sex inference metrics:

```
%%bash
# Download microarray PLINK data if not present
if [ ! -f "../../arrays.bed" ]; then
    echo "Downloading microarray data..."
    gsutil -u "$GOOGLE_PROJECT" -m cp -r gs://fc-aou-datasets-controlled/v8/microarray/plink/* ../..
fi
```

Build `gnomon`:
```
%%bash
# Install Rust nightly
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && { for f in ~/.bashrc ~/.profile; do [ -f "$f" ] || touch "$f"; grep -qxF 'source "$HOME/.cargo/env"' "$f" || printf '\n# Rust / Cargo\nsource "$HOME/.cargo/env"\n' >> "$f"; done; } && source "$HOME/.cargo/env" && rustup toolchain install nightly && rustup default nightly
git clone https://github.com/SauersML/gnomon.git
cd gnomon
rustup override set nightly
cargo build --release
```

Run Gnomon sex inference to generate arrays.sex.tsv with metric columns:
```
!echo "Running Gnomon sex inference..."
!./gnomon/target/release/gnomon terms --sex ../../arrays.bed
```

Load data and visualize specific inference metrics against Self-Reported Sex:

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC

# =============================================================================
# 1. PREPARE DATA FOR SVM
# =============================================================================
# Filter to clean Male/Female data for training the separator
train_df = df.dropna(subset=['sex_at_birth', 'X_AutoHet_Ratio', 'Y_Density']).copy()
train_df = train_df[train_df['sex_at_birth'].isin(['Male', 'Female'])]

# Features (X) and Target (y)
X = train_df[['X_AutoHet_Ratio', 'Y_Density']].values
# Map labels: Male=1, Female=0
y = (train_df['sex_at_birth'] == 'Male').astype(int).values

# =============================================================================
# 2. TRAIN LINEAR SVM (Find the Best Line)
# =============================================================================
# C=1.0 is standard regularization. kernel='linear' forces a straight line.
clf = SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# =============================================================================
# 3. EXTRACT THE LINE EQUATION
# =============================================================================
# The equation of the separating plane is: w0*x + w1*y + b = 0
w = clf.coef_[0]
b = clf.intercept_[0]

# To plot as y = mx + c:
# w0*x + w1*y + b = 0  =>  y = -(w0/w1)*x - (b/w1)
slope = -w[0] / w[1]
intercept = -b / w[1]

print("="*60)
print("OPTIMAL SEPARATION LINE EQUATION")
print("="*60)
print(f"Formula: Y_Density = ({slope:.4f} * X_AutoHet_Ratio) + {intercept:.4f}")
print("-" * 60)
print("Decision Logic:")
print(f"If Point is ABOVE line -> MALE")
print(f"If Point is BELOW line -> FEMALE")

# =============================================================================
# 4. VISUALIZATION
# =============================================================================
plt.figure(figsize=(12, 9))
sns.set_theme(style="whitegrid")

# 1. Plot the actual Data Points
sns.scatterplot(
    data=train_df, 
    x='X_AutoHet_Ratio', 
    y='Y_Density', 
    hue='sex_at_birth', 
    palette={'Male': '#1f77b4', 'Female': '#e377c2'},
    alpha=0.6,
    s=50,
    edgecolor='k'
)

# 2. Plot the SVM Decision Boundary (The Optimal Line)
# Create a range of x values across the plot
x_vals = np.linspace(train_df['X_AutoHet_Ratio'].min(), train_df['X_AutoHet_Ratio'].max(), 100)
y_vals = slope * x_vals + intercept

plt.plot(x_vals, y_vals, color='black', linewidth=3, linestyle='-', label='SVM Optimal Separator')

# 3. Plot the Margins (The "Street" width)
# The margins are 1 unit away in the transformed space
margin = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
y_margin_up = y_vals + np.sqrt(1 + slope**2) * margin
y_margin_down = y_vals - np.sqrt(1 + slope**2) * margin

plt.plot(x_vals, y_margin_up, 'k--', linewidth=1, alpha=0.5, label='Maximum Margin')
plt.plot(x_vals, y_margin_down, 'k--', linewidth=1, alpha=0.5)

# Fill the areas to show classification zones
plt.fill_between(x_vals, y_vals, 2.0, color='#1f77b4', alpha=0.1) # Male Zone
plt.fill_between(x_vals, -1.0, y_vals, color='#e377c2', alpha=0.1) # Female Zone

# Formatting
plt.title(f"SVM Optimal Decision Boundary (Accuracy: {clf.score(X, y)*100:.2f}%)", fontsize=16, fontweight='bold')
plt.xlabel("X-to-Autosome Heterozygosity Ratio", fontsize=12, fontweight='bold')
plt.ylabel("Y-Genome Density", fontsize=12, fontweight='bold')

# Lock axis to relevant data range
plt.ylim(-0.1, 1.2)
plt.xlim(-0.1, 1.4)

plt.legend(loc='lower left', frameon=True, facecolor='white', framealpha=1)
plt.tight_layout()
plt.show()
```
