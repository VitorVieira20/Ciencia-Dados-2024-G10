from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import r2_score
from tensorflow.keras import layers
from tensorflow import keras

class ModelE:
    def __init__(self, data_train, labels_train, data_test, labels_test):
        self.data_train = data_train
        self.labels_train = labels_train
        self.data_test = data_test
        self.labels_test = labels_test
        self.X_train_normalized = None
        self.X_test_normalized = None

        mean = self.data_train.mean(axis=0)
        std = self.data_train.std(axis=0)
        self.X_train_normalized = (self.data_train - mean) / std
        self.X_test_normalized = (self.data_test - mean) / std

        self.model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=[self.data_train.shape[1]]),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(64, activation='relu'),
            layers.Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

    def train_and_evaluate(self):

        history = self.model.fit(self.X_train_normalized, self.labels_train, epochs=20, batch_size=32, validation_split=0.2)

        test_loss, test_mse = self.model.evaluate(self.X_test_normalized, self.labels_test)

        predictions = self.model.predict(self.X_test_normalized)

        r2 = r2_score(self.labels_test, predictions)

        print("Test Loss:", test_loss)
        print("Test MSE:", test_mse)
        print("Test R^2 Score:", r2)

        return self.model