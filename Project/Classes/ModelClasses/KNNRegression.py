import numpy as np
from sklearn.metrics import r2_score


class KNNRegression:
    """
    Implementation of the K-Nearest Neighbors (KNN) algorithm for regression.

    Attributes:
        k (int): Number of neighbors to consider for regression.

    Methods:
        __init__(self, k=3): Initializes the KNN regressor with the specified value of k.
        fit(self, X, y): Fits the KNN regressor to the training data.
        predict(self, X): Predicts the target values for the input data.
        _predict(self, x): Predicts the target value for a single data point.
        euclidean_distance(x1, x2): Computes the Euclidean distance between two data points.
        evaluate(self, X_test, y_test): Calculates and prints the R² score of the model.
    """

    def __init__(self, k=3):
        """
        Initializes the KNN regressor with the specified value of k.

        Args:
            k (int, optional): Number of neighbors to consider for regression. Default is 3.
        """
        self.k = k

    def fit(self, X, y):
        """
        Fits the KNN regressor to the training data.

        Args:
            X (array-like): Training data features.
            y (array-like): Training data target values.
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
        Predicts the target values for the input data.

        Args:
            X (DataFrame): Input data features.

        Returns:
            list: Predicted target values.
        """
        predictions = [self._predict(x) for x in X.values]
        return predictions

    def _predict(self, x):
        """
        Predicts the target value for a single data point.

        Args:
            x (array-like): Input data point.

        Returns:
            float: Predicted target value.
        """
        # compute the distance
        distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train.values]

        # get the closest k
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_targets = self.y_train[k_indices]

        # return the average of the target values of the k nearest neighbors
        return np.mean(k_nearest_targets)

    def evaluate(self, X_test, y_test):
        """
        Calculates and prints the R² score of the model.

        Args:
            X_test (array-like): Test data features.
            y_test (array-like): Test data target values.
        """

        print("\nKNN Model fit and evaluation...")

        # Making predictions on test data
        predictions = self.predict(X_test)

        # Calculating R² score
        r2 = r2_score(y_test, predictions)


        print("KNN Regression Score:", r2)
        print("-----------------------------------------------------\n")
        return r2
