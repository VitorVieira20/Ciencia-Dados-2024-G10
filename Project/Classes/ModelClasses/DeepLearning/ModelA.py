from tensorflow.keras import models, layers

class ModelA:
    def __init__(self):
        self.model = models.Sequential([
            layers.Dense(100, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(50, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error', 'r2_score'])

    def train_and_evaluate(self, train_data, train_labels, test_data, test_labels):

        history = self.model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2)

        test_loss, test_mse, test_r2 = self.model.evaluate(test_data, test_labels)
        print("Test MSE:", test_loss)
        print("Test R^2:", test_r2)

        return self.model