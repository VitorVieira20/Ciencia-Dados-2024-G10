import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class KmeansClustering:
    def see_optimize_k(self, data):
        """
        Perform KMeans clustering on the input data and visualize the optimization of the number of clusters (k).

        Parameters:
        - data: Input data for clustering

        Returns:
        - None
        """
        print("\nKMeans Clustering...")

        # Extracting relevant features from the input data
        X = data[['bath', 'rooms', 'room_bath_ratio', 'room_balcony_ratio']]

        # Remove rows with NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = X[~np.isnan(X).any(axis=1)]
            X = X[~np.isinf(X).any(axis=1)]

        # Standardize the features
        scaler = StandardScaler()
        X_Normalized = scaler.fit_transform(X)

        # Function to optimize the number of clusters (k) using Elbow method
        def optimize_k_means(data, max_k):
            means = []
            inertias = []

            for k in range(1, max_k):
                kmeans = KMeans(n_clusters=k)
                kmeans.fit(data)

                means.append(k)
                inertias.append(kmeans.inertia_)

            # Plotting Elbow curve
            plt.subplots(figsize=(10, 5))
            plt.plot(means, inertias, 'o-')
            plt.xlabel('Clusters')
            plt.ylabel('Inertia')
            plt.grid(True)
            plt.show()

        # Call the optimization function
        optimize_k_means(X_Normalized, 15)

        # Define pairs of features for visualization
        feature_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        custom_colors = ['blue', 'green', 'orange', 'purple']

        # Plot clusters for different feature pairs
        plt.figure(figsize=(15, 10))
        for i, (feature1, feature2) in enumerate(feature_pairs, 1):
            plt.subplot(2, 3, i)

            X_pair = X_Normalized[:, [feature1, feature2]]

            # Fit KMeans model and visualize clusters
            kmeans = KMeans(n_clusters=4, random_state=42)
            labels = kmeans.fit_predict(X_pair)

            plt.scatter(X_pair[:, 0], X_pair[:, 1], c=[custom_colors[label] for label in labels], edgecolors='k', s=60)

            centers = kmeans.cluster_centers_
            plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=200, label='Centroids')
            plt.xlabel(f'Feature {feature1}')
            plt.ylabel(f'Feature {feature2}')
            plt.title(f'Clusters for Features {feature1} and {feature2}')
            plt.legend()

        # Plot Elbow curve for each feature pair
        plt.figure(figsize=(15, 5))
        for i, (feature1, feature2) in enumerate(feature_pairs, 1):
            X_pair = X_Normalized[:, [feature1, feature2]]
            distortions = []
            silhouette_scores = []
            for k in range(2, 11):
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(X_pair)
                distortions.append(kmeans.inertia_)
                silhouette_scores.append(silhouette_score(X_pair, kmeans.labels_))

            # Plot the Elbow curve
            plt.subplot(2, 3, i)
            plt.plot(range(2, 11), distortions, marker='o')
            plt.title(f'Elbow Curve for Features {feature1} and {feature2}')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Distortion')
            plt.grid(True)

            # Print Silhouette Scores for each feature pair
            print(f'Silhouette Scores for Features {feature1} and {feature2}: {silhouette_scores}')