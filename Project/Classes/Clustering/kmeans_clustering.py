import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

class KmeansClustering:
    def see_k_means(self, data_train, data_test):
        """
        Perform KMeans clustering on the input data and visualize the optimization of the number of clusters (k).

        Parameters:
        - data_train: Data train for clustering
        - data_test: Data test for clustering

        Returns:
        - None
        """
        print("\nKMeans Clustering...")

        # Call the optimization function
        self._optimize_k_means(data_train, 20)

        # See what's the ideal pca components number
        self._pca_components(data_train)

        # Perform actual K-Means
        kmeans = KMeans(n_clusters=7, random_state=42)
        components = 3
        kmeans.fit(data_train)

        X_test_labels = kmeans.predict(data_test)

        pca = PCA(components)
        data_pca = pca.fit_transform(data_test)

        centroids_pca = pca.transform(kmeans.cluster_centers_)

        data_pca_df = pd.DataFrame(data_pca, columns=[f'PC{i + 1}' for i in range(components)])
        data_pca_df['Cluster'] = X_test_labels

        pairplot = sns.pairplot(data_pca_df, hue='Cluster', palette='viridis', plot_kws={'s': 20})

        for ax in pairplot.axes.flatten():
            lim = ax.get_xlim()
            for centroid in centroids_pca:
                ax.plot(centroid[0], centroid[1], 'rx', markersize=5, mew=2)
            ax.set_xlim(lim)

        plt.suptitle("Pairplot of Clusters Using First Three Principal Components", y=1.02)
        plt.show()

    def _optimize_k_means(self, data, max_k):
        means = []
        inertias = []

        for k in range(1, max_k):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(data)

            means.append(k)
            inertias.append(kmeans.inertia_)

        plt.subplots(figsize=(10, 5))
        plt.plot(means, inertias, 'o-')
        plt.xlabel('Clusters')
        plt.ylabel('Inertia')
        plt.grid(True)
        plt.show()

    def _pca_components(self, data):
        pca = PCA()
        pca.fit(data)

        explained_variance_ratio_cumulative = np.cumsum(pca.explained_variance_ratio_)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(explained_variance_ratio_cumulative) + 1), explained_variance_ratio_cumulative,
                 marker='o', linestyle='--')
        plt.xlabel('PCA Components')
        plt.ylabel('Accumulated Explained Variance')
        plt.title('Accumulated Explained Variance by Number of Principal Components')
        plt.grid(True)
        plt.show()