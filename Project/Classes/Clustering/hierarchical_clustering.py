import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


class HierarchicalClustering:
    def see_cluster_plots(self, data):
        """
        Perform hierarchical clustering on the data and visualize dendrograms using different linkage methods.

        Parameters:
        - data: Input data for clustering

        Returns:
        - None
        """
        print("\nHierarchical Clustering...")

        X = data.values

        # Remove rows with NaN or infinite values
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = X[~np.isnan(X).any(axis=1)]
            X = X[~np.isinf(X).any(axis=1)]

        # Define different linkage methods
        linkage_methods = ['ward', 'complete', 'average', 'single']

        # Plot dendrograms for each linkage method
        plt.figure(figsize=(15, 10))
        for i, method in enumerate(linkage_methods, 1):
            plt.subplot(2, 2, i)
            plt.title(f'Dendrogram ({method} linkage)')
            dendrogram(linkage(X, method=method))
            plt.xlabel('Samples')
            plt.ylabel('Euclidean distances')
        plt.tight_layout()
        plt.show()