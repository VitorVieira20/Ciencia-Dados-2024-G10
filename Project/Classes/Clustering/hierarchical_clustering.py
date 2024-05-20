import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch


class HierarchicalClustering:
    def see_cluster_plots(self, data):
        """
        Perform hierarchical clustering on the data and visualize dendrograms using different linkage methods.

        Parameters:
        - data: Data for clustering

        Returns:
        - None
        """
        print("\nHierarchical Clustering...")

        # Define different linkage methods
        linkage_methods = ['ward', 'complete', 'average', 'single']

        plt.figure(figsize=(15, 10))
        for i, method in enumerate(linkage_methods, 1):
            plt.subplot(2, 2, i)
            plt.title(f'Dendrogram ({method} linkage)')
            sch.dendrogram(sch.linkage(data, method=method), color_threshold=30)
            plt.xlabel('Samples')
            plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()