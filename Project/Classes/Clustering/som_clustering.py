import numpy as np
from matplotlib import pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class SOMClustering:
    def see_som_clustering(self, data):
        """
        Perform Self-Organizing Map (SOM) clustering on the input data and visualize the results.

        Parameters:
        - data: Input data for clustering

        Returns:
        - None
        """
        print("\nSOM Clustering...")

        X = data[['bath', 'rooms', 'room_bath_ratio', 'room_balcony_ratio']]

        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            X = X[~np.isnan(X).any(axis=1)]
            X = X[~np.isinf(X).any(axis=1)]

        scaler = StandardScaler()
        X_Normalized = scaler.fit_transform(X)

        som_size = (10, 10)

        num_features = X_Normalized.shape[1]

        som = MiniSom(som_size[0], som_size[1], num_features, sigma=0.5, learning_rate=0.5)

        som.random_weights_init(X_Normalized)

        num_epochs = 1000
        som.train_random(X_Normalized, num_epochs)

        som_weights = som.get_weights()

        winners = np.array([som.winner(x) for x in X_Normalized])

        kmeans = KMeans(n_clusters=4, random_state=42)
        cluster_labels = kmeans.fit_predict(winners)

        plt.figure(figsize=(10, 10))
        plt.pcolor(som.distance_map().T, cmap='bone_r', alpha=0.5)
        plt.colorbar()

        for i, (x, label) in enumerate(zip(X_Normalized, cluster_labels)):
            w = winners[i]
            plt.scatter(w[0] + 0.5, w[1] + 0.5, c=plt.cm.tab10(label), edgecolors='k', s=100)

        plt.title('SOM Clustering')
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.pcolor(som_weights[:, :, 0].T, cmap='viridis')
        plt.colorbar()
        plt.title('Heatmap of Neuron Weights')
        plt.show()