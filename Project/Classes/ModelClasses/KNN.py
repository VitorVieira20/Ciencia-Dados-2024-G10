import numpy as np

class KNN:
    """
    Implementation of the K-Nearest Neighbors (KNN) algorithm for classification.

    Attributes:
        k (int): Number of neighbors to consider for classification.

    Methods:
        __init__(self, k=3): Initializes the KNN classifier with the specified value of k.
        fit(self, X, y): Fits the KNN classifier to the training data.
        predict(self, X): Predicts the class labels for the input data.
        _predict(self, x): Predicts the class label for a single data point.
        euclidean_distance(x1, x2): Computes the Euclidean distance between two data points.
    """

    def __init__(self, k=3):
        """
        Initializes the KNN classifier with the specified value of k.

        Args:
            k (int, optional): Number of neighbors to consider for classification. Default is 3.
        """
        self.k = k

    def fit(self, X, y):
        """
        Fits the KNN classifier to the training data.

        Args:
            X (array-like): Training data features.
            y (array-like): Training data labels.
        """
        self.X_train = X
        self.y_train = y

    @staticmethod
    def euclidean_distance(x1, x2):
        """
        Computes the Euclidean distance between two data points.

        Args:
            x1 (array-like): First data point.
            x2 (array-like): Second data point.

        Returns:
            float: Euclidean distance between x1 and x2.
        """
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
        return distance

    def predict(self, X):
        """
        Predicts the class labels for the input data.

        Args:
            X (DataFrame): Input data features.

        Returns:
            list: Predicted class labels.
        """
        predictions = [self._predict(x) for x in X.values]
        return predictions

    def _predict(self, x):
        """
        Predicts the class label for a single data point.

        Args:
            x (array-like): Input data point.

        Returns:
            int: Predicted class label.
        """
        # compute the distance
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train.values]

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_indices]

        # count occurrences of each class label
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)

        # find the index of the most frequent class label
        most_common_index = np.argmax(counts)

        # return the most common class label
        return unique_labels[most_common_index]

    def evaluate(self, X_test, y_test):
        """
        Calculates and prints the accuracy of the model.

        Args:
            X_test (array-like): Test data features.
            y_test (array-like): Test data labels.
        """
        # Making predictions on test data
        predictions = self.predict(X_test)

        # Calculating accuracy
        correct_predictions = np.sum(predictions == y_test)
        total_predictions = len(y_test)
        accuracy = correct_predictions / total_predictions

        print("KNN Model Accuracy:", accuracy)