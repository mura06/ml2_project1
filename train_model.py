import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from minisom import MiniSom
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# Import the preprocessing functions from feature_engineering
from feature_engineering import preprocess_data, create_new_features, encode_categorical_features

def perform_clustering(df, features, n_clusters=6):
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

def perform_som(df, features, x=3, y=3, sigma=1.0, learning_rate=0.5, num_iteration=1000):
    """
    Perform Self-Organizing Maps (SOM) clustering on the given features.
    """
    # Create a copy to avoid SettingWithCopyWarning
    clustering_data = df[features].copy()
    
    # Handle missing values by filling with median
    clustering_data = clustering_data.fillna(clustering_data.median())
    
    # Scale the features
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(clustering_data)
    
    # Fit SOM
    som = MiniSom(x=x, y=y, input_len=scaled_data.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed=42)
    som.random_weights_init(scaled_data)
    som.train_random(data=scaled_data, num_iteration=num_iteration)
    
    # Get cluster labels by mapping each point to its Best Matching Unit (BMU)
    # Convert the 2D grid coordinates (i, j) into a 1D cluster label
    cluster_labels = np.array([som.winner(d)[0] * y + som.winner(d)[1] for d in scaled_data])
    
    return cluster_labels, scaled_data, som

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
    # plt.show()

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
    # plt.show()

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
    # plt.show()

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
    
    # Based on the elbow method, we'll choose an optimal k=6
    k = 6
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
    
    # Perform SOM clustering
    print("\nPerforming SOM clustering...")
    # 3x3 grid = 9 clusters
    som_x, som_y = 3, 3
    som_labels, _, som_model = perform_som(df, cluster_features, x=som_x, y=som_y)
    
    # Add SOM cluster labels to the original dataframe
    df['SOM_Cluster'] = som_labels
    
    # Calculate Silhouette Score for SOM
    if len(set(som_labels)) > 1:
        som_silhouette = silhouette_score(scaled_data, som_labels)
        print(f"SOM Silhouette Score: {som_silhouette:.4f}")
    else:
        print("SOM Silhouette Score: Not enough valid clusters formed.")
    
    # Visualize SOM clusters using t-SNE
    print("Visualizing SOM clusters using t-SNE...")
    plot_clusters_tsne(scaled_data, som_labels, title='SOM Clusters Visualized with t-SNE')
    
    # Analyze K-Means clusters
    print("\n--- K-Means Cluster Analysis (Mean values for each feature per cluster) ---")
    kmeans_analysis = df.groupby('KMeans_Cluster')[cluster_features].mean().round(2)
    print(kmeans_analysis)
    
    print("\nExample games from each K-Means cluster:")
    for i in range(k):
        print(f"\nKMeans Cluster {i}:")
        sample_games = df[df['KMeans_Cluster'] == i]['name'].head(5).tolist()
        print(", ".join(str(name) for name in sample_games))

    # Analyze SOM clusters
    print("\n--- SOM Cluster Analysis (Mean values for each feature per cluster) ---")
    som_analysis = df.groupby('SOM_Cluster')[cluster_features].mean().round(2)
    print(som_analysis)
    
    print("\nExample games from each SOM cluster:")
    unique_som_labels = sorted(df['SOM_Cluster'].unique())
    for label in unique_som_labels:
        print(f"\nSOM Cluster {label}:")
        sample_games = df[df['SOM_Cluster'] == label]['name'].head(5).tolist()
        print(", ".join(str(name) for name in sample_games))

    # Save the clustered dataframe for the Streamlit app
    print("\nSaving clustered dataset to 'clustered_games.csv'...")
    df.to_csv("clustered_games.csv", index=False)
    print("Done!")
