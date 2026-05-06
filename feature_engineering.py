import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --- Preprocessing & Feature Engineering ---

def preprocess_data(data):
    """Basic data cleaning and preprocessing."""
    # Drop duplicate rows
    data = data.drop_duplicates()
    
    # metacritic_score has ~64% missing values, so it's better to drop it
    if 'metacritic_score' in data.columns:
        data = data.drop(columns=['metacritic_score'])
    if 'recommendations' in data.columns:
        data['recommendations'] = data['recommendations'].fillna(0)
        
    # Convert release_date to datetime format
    if 'release_date' in data.columns:
        data['release_date'] = pd.to_datetime(data['release_date'], format='mixed', errors='coerce')
        
    return data

def create_new_features(data):
    """Create new relevant features from existing ones."""
    # Total Reviews
    if 'positive_reviews' in data.columns and 'negative_reviews' in data.columns:
        data['total_reviews'] = data['positive_reviews'] + data['negative_reviews']
        
        # Review Sentiment (Positive Review Ratio)
        data['positive_review_ratio'] = np.where(data['total_reviews'] > 0, 
                                               data['positive_reviews'] / data['total_reviews'], 
                                               0)
    
    # Release Year and Month
    if 'release_date' in data.columns:
        data['release_year'] = data['release_date'].dt.year
        data['release_month'] = data['release_date'].dt.month
        
    # Number of Platforms
    platforms = ['platforms_win', 'platforms_mac', 'platforms_linux']
    if all(col in data.columns for col in platforms):
        data['num_platforms'] = data['platforms_win'].astype(int) + \
                              data['platforms_mac'].astype(int) + \
                              data['platforms_linux'].astype(int)
                              
    # Average Estimated Owners (parsing the string range)
    def parse_owners(owner_str):
        if pd.isna(owner_str):
            return 0
        try:
            parts = owner_str.replace(',', '').split(' .. ')
            return (int(parts[0]) + int(parts[1])) / 2
        except:
            return 0
            
    if 'estimated_owners' in data.columns:
        data['avg_estimated_owners'] = data['estimated_owners'].apply(parse_owners)
        
    return data

def encode_categorical_features(data):
    """One-hot encode categorical features like genres."""
    if 'genres' in data.columns:
        # Fill NA with empty string
        genres_clean = data['genres'].fillna('')
        # Get dummies for comma-separated genres
        genre_dummies = genres_clean.str.get_dummies(sep=', ')
        # Add a prefix to clearly identify genre columns
        genre_dummies = genre_dummies.add_prefix('genre_')
        # Join with original data
        data = pd.concat([data, genre_dummies], axis=1)
        
    return data

if __name__ == "__main__":
    df = pd.read_csv("steam_top_games_2026.csv")

    # --- Data Vizualization ---

    print(df.head())
    print(df.shape) # Check the number of rows and columns
    print(df.columns) # Check the column names
    print(df.info()) # Check for missing values and data types
    print(df.describe().round(2)) # Check main statistics of the data

    # Categorical variables
    df_c = df.select_dtypes(include=['object'])
    print(df_c.head(), "\n")

    # Look for inconsistencies in the categorical variables
    for i in df_c.columns:
        print(df_c[i].value_counts(), "\n")


    # Numerical variables
    df_n = df.select_dtypes(include=['int64', 'float64'])
    print(df_n.head())


    # Outliers detection with IQR method (boxplots)
    outliers = {}

    for col in df_n.columns:
        Q1 = df_n[col].quantile(0.25)
        Q3 = df_n[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers[col] = df_n[(df_n[col] < lower_bound) | (df_n[col] > upper_bound)][col]

    # Visualize the distribution of numerical variables with boxplots
    n_cols = 3
    n_rows = (len(df_n.columns) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3))
    axes = axes.flatten()

    for i, col in enumerate(df_n.columns):    
        sns.boxplot(x=df_n[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()

    # Correlation matrix
    corr_matrix = df_n.corr(method='pearson')

    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title("Correlation Heatmap")
    plt.show()

    # Apply preprocessing
    print("\n--- Applying Preprocessing ---")
    df = preprocess_data(df)
    print("Data shape after preprocessing:", df.shape)

    # Apply feature engineering
    print("\n--- Applying Feature Engineering ---")
    df = create_new_features(df)

    # Encode categorical features
    print("\n--- Encoding Categorical Features ---")
    df = encode_categorical_features(df)

    # Check the new features
    new_cols = ['total_reviews', 'positive_review_ratio', 'release_year', 'num_platforms', 'avg_estimated_owners']
    available_new_cols = [col for col in new_cols if col in df.columns]
    print(f"Created new features: {available_new_cols}")
    print(df[available_new_cols].head())

