import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

class HierarchicalClustering:
    def see_cluster_plots(self, data):
        print("\nHierarchical Clustering...")

        X = data.values

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = X[~np.isnan(X).any(axis=1)]
            X = X[~np.isinf(X).any(axis=1)]

        linkage_methods = ['ward', 'complete', 'average', 'single']

        plt.figure(figsize=(15, 10))
        for i, method in enumerate(linkage_methods, 1):
            plt.subplot(2, 2, i)
            plt.title(f'Dendrogram ({method} linkage)')
            dendrogram(linkage(X, method=method))
            plt.xlabel('Samples')
            plt.ylabel('Euclidian distances')
        plt.tight_layout()
        plt.show()
