from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class ModelC():
    def __init__(self, data_train, labels_train, data_test, labels_test):
        self.data_train = data_train
        self.labels_train = labels_train
        self.data_test = data_test
        self.labels_test = labels_test
        self.X_train_normalized = None
        self.X_test_normalized = None

        scaler = StandardScaler()
        self.X_train_normalized = scaler.fit_transform(self.data_train)
        self.X_test_normalized = scaler.transform(self.data_test)

        self.model = Sequential([
            Dense(128, activation='relu', input_shape=(self.X_train_normalized.shape[1],)),
            Dense(64, activation='relu'),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])


    def train_and_evaluate(self):
        history = self.model.fit(self.X_train_normalized, self.labels_train, epochs=10, batch_size=32)

        test_loss, test_mse = self.model.evaluate(self.X_test_normalized, self.labels_test)

        predictions = self.model.predict(self.X_test_normalized)

        test_r2_score = r2_score(self.labels_test, predictions)

        print("Test Loss:", test_loss)
        print("Test MSE:", test_mse)
        print("Test R^2 Score:", test_r2_score)

        return self.model