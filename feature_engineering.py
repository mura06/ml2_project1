import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("steam_top_games_2026.csv")

# checking the data content and structure

print(df.head())
print(df.shape) #check the number of rows and columns
print(df.columns) #check the column names
print(df.info()) #check for missing values and data types
print(df.describe().round(2)) # check main statistics of the data

# categorical variables
df_c = df.select_dtypes(include=['object'])
print(df_c.head(), "\n")

# look for inconsistencies in the categorical variables
for i in df_c.columns:
    print(df_c[i].value_counts(), "\n")


# numerical variables
df_n = df.select_dtypes(include=['int64', 'float64'])
print(df_n.head())


# outliers detection with IQR method (boxplots)
outliers = {}

for col in df_n.columns:
    Q1 = df_n[col].quantile(0.25)
    Q3 = df_n[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers[col] = df_n[(df_n[col] < lower_bound) | (df_n[col] > upper_bound)][col]

# visualize the distribution of numerical variables with boxplots
n_cols = 3
n_rows = (len(df_n.columns) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
axes = axes.flatten()

for i, col in enumerate(df_n.columns):    
    sns.boxplot(x=df_n[col], ax=axes[i])
    axes[i].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()



# correlation matrix
corr_matrix = df_n.corr(method='pearson')

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title("Correlation Heatmap")
plt.show()



