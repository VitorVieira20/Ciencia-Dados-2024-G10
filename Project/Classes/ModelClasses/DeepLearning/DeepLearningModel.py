from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

class NeuralNetworkRegressionModel:
    def fit_and_evaluate(self, train_data, train_labels, test_data, test_labels):
        """
        Fits and evaluates a Neural Network Regression model.

        Parameters:
        - train_data: Training data
        - train_labels: Labels for training data
        - test_data: Test data
        - test_labels: Labels for test data

        Returns:
        - model: Best trained Neural Network Regression model
        - history: Training history of the model
        """

        # Data normalization
        scaler = StandardScaler()
        X_train_normalized = scaler.fit_transform(train_data)
        X_test_normalized = scaler.transform(test_data)

        # Neural Network Model
        print("Neural Network Regression Model fit and evaluation...")

        # Define the model architecture
        model = Sequential([
            Dense(512, activation='relu', input_shape=(X_train_normalized.shape[1],)),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(1)
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

        # Fit the model
        history = model.fit(X_train_normalized, train_labels, epochs=10, batch_size=32, verbose=0)

        # Evaluate the model on test data
        test_loss, test_mse = model.evaluate(X_test_normalized, test_labels, verbose=0)

        # Make predictions on test data
        predictions = model.predict(X_test_normalized)

        # Calculate R-squared score
        test_r2_score = r2_score(test_labels, predictions)

        # Print evaluation metrics
        print("Test Loss:", test_loss)
        print("Test MSE:", test_mse)
        print("Test R^2 Score:", test_r2_score)
        print("-----------------------------------------------------\n")

        # Plot training history
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

        return model, history