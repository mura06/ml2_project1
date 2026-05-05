import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Import the preprocessing functions from feature_engineering
from feature_engineering import preprocess_data, create_new_features, encode_categorical_features

def perform_clustering(df, features, n_clusters=4):
    """
    Perform K-Means clustering on the given features.
    """
    # Create a copy to avoid SettingWithCopyWarning
    clustering_data = df[features].copy()
    
    # Handle missing values by filling with median
    clustering_data = clustering_data.fillna(clustering_data.median())
    
    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    return cluster_labels, scaled_data, kmeans

def perform_dbscan(df, features, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering on the given features.
    """
    # Create a copy to avoid SettingWithCopyWarning
    clustering_data = df[features].copy()
    
    # Handle missing values by filling with median
    clustering_data = clustering_data.fillna(clustering_data.median())
    
    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    # Fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(scaled_data)
    
    return cluster_labels, scaled_data, dbscan

def plot_elbow_method(scaled_data, max_k=10):
    """
    Plot the elbow method to help determine optimal k.
    """
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)
        
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='--')
    plt.title('Elbow Method For Optimal k')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.xticks(range(1, max_k + 1))
    plt.grid(True)
    plt.show()

def plot_clusters_pca(scaled_data, cluster_labels, title='Clusters Visualized with PCA'):
    """
    Reduce dimensions to 2D using PCA and plot the clusters.
    """
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_data)
    
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='PC1', y='PC2', hue='Cluster', palette='viridis', data=pca_df, alpha=0.7)
    plt.title(title)
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.show()

def plot_clusters_tsne(scaled_data, cluster_labels, title='Clusters Visualized with t-SNE'):
    """
    Reduce dimensions to 2D using t-SNE and plot the clusters.
    """
    tsne = TSNE(n_components=2, random_state=42)
    tsne_components = tsne.fit_transform(scaled_data)
    
    tsne_df = pd.DataFrame(data=tsne_components, columns=['t-SNE 1', 't-SNE 2'])
    tsne_df['Cluster'] = cluster_labels
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='t-SNE 1', y='t-SNE 2', hue='Cluster', palette='viridis', data=tsne_df, alpha=0.7)
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = pd.read_csv("steam_top_games_2026.csv")
    df = preprocess_data(df)
    df = create_new_features(df)
    df = encode_categorical_features(df)
    
    # Select features for clustering
    # We want features that represent game characteristics (popularity, price, playtime, etc.)
    base_features = [
        'price_usd', 
        'avg_playtime_forever', 
        'peak_ccu', 
        'total_reviews', 
        'positive_review_ratio',
        'avg_estimated_owners',
        'dlc_count'
    ]
    
    # Add genre one-hot encoded columns to the clustering features
    genre_features = [col for col in df.columns if col.startswith('genre_')]
    cluster_features = base_features + genre_features
    
    # Keep only features that are present in the dataframe
    cluster_features = [f for f in cluster_features if f in df.columns]
    print(f"Features used for clustering: {len(cluster_features)} features (including genres)")
    
    # Temporarily prepare scaled data to find elbow
    temp_data = df[cluster_features].copy()
    temp_data = temp_data.fillna(temp_data.median())
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(temp_data)
    
    # Plot elbow method
    print("Plotting elbow method...")
    plot_elbow_method(scaled_data, max_k=10)
    
    # Based on general datasets, we'll choose a reasonable k=4
    k = 4
    print(f"\nPerforming K-Means clustering with k={k}...")
    kmeans_labels, scaled_data, kmeans_model = perform_clustering(df, cluster_features, n_clusters=k)
    
    # Add K-Means cluster labels to the original dataframe
    df['KMeans_Cluster'] = kmeans_labels
    
    # Calculate Silhouette Score for K-Means
    if len(set(kmeans_labels)) > 1:
        kmeans_silhouette = silhouette_score(scaled_data, kmeans_labels)
        print(f"K-Means Silhouette Score: {kmeans_silhouette:.4f}")
    
    # Visualize K-Means clusters using t-SNE
    print("Visualizing K-Means clusters using t-SNE...")
    plot_clusters_tsne(scaled_data, kmeans_labels, title='K-Means Clusters Visualized with t-SNE')
    
    # Perform DBSCAN clustering
    print("\nPerforming DBSCAN clustering...")
    # eps and min_samples might need tuning depending on the dataset characteristics
    # Note: Adding categorical features increases dimensionality significantly, which might require a larger eps
    dbscan_labels, _, dbscan_model = perform_dbscan(df, cluster_features, eps=2.0, min_samples=5)
    
    # Add DBSCAN cluster labels to the original dataframe
    df['DBSCAN_Cluster'] = dbscan_labels
    
    # Calculate Silhouette Score for DBSCAN (ignoring noise points labeled as -1)
    dbscan_clustered_indices = np.where(dbscan_labels != -1)[0]
    if len(set(dbscan_labels[dbscan_clustered_indices])) > 1:
        dbscan_silhouette = silhouette_score(
            scaled_data[dbscan_clustered_indices], 
            dbscan_labels[dbscan_clustered_indices]
        )
        print(f"DBSCAN Silhouette Score (excluding noise): {dbscan_silhouette:.4f}")
    else:
        print("DBSCAN Silhouette Score: Not enough valid clusters formed (mostly noise).")
    
    # Visualize DBSCAN clusters using t-SNE
    print("Visualizing DBSCAN clusters using t-SNE...")
    plot_clusters_tsne(scaled_data, dbscan_labels, title='DBSCAN Clusters Visualized with t-SNE')
    
    # Analyze K-Means clusters
    print("\n--- K-Means Cluster Analysis (Mean values for each feature per cluster) ---")
    kmeans_analysis = df.groupby('KMeans_Cluster')[cluster_features].mean().round(2)
    print(kmeans_analysis)
    
    print("\nExample games from each K-Means cluster:")
    for i in range(k):
        print(f"\nKMeans Cluster {i}:")
        sample_games = df[df['KMeans_Cluster'] == i]['name'].head(5).tolist()
        print(", ".join(str(name) for name in sample_games))

    # Analyze DBSCAN clusters
    print("\n--- DBSCAN Cluster Analysis (Mean values for each feature per cluster) ---")
    # DBSCAN often returns -1 for noise points/outliers
    dbscan_analysis = df.groupby('DBSCAN_Cluster')[cluster_features].mean().round(2)
    print(dbscan_analysis)
    
    print("\nExample games from each DBSCAN cluster (-1 represents noise/outliers):")
    unique_dbscan_labels = sorted(df['DBSCAN_Cluster'].unique())
    for label in unique_dbscan_labels:
        print(f"\nDBSCAN Cluster {label}:")
        sample_games = df[df['DBSCAN_Cluster'] == label]['name'].head(5).tolist()
        print(", ".join(str(name) for name in sample_games))
