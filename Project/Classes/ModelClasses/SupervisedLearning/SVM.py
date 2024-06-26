import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR

# Importing custom functions from shared_functions module
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot, models_training_loss_plot

class SVMModel:
    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        """
        Fits and evaluates a Support Vector Machine (SVM) model.

        Parameters:
        - data_train: Training data
        - labels_train: Labels for training data
        - data_test: Test data
        - labels_test: Labels for test data

        Returns:
        - best_svm_model: Best trained SVM model
        """

        # SVM Model
        print("SVM Model fit and evaluation...")

        # Hyperparameters grid for GridSearchCV
        hyperparameters = [
            {'kernel': ['linear'], 'C': [0.1, 1, 10]},
            {'kernel': ['rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
            {'kernel': ['poly'], 'C': [0.1, 1, 10], 'degree': [2, 3]},
            {'kernel': ['sigmoid'], 'C': [0.1, 1, 10]}
        ]

        # Initializing SVM model
        svm_model = SVR()

        # GridSearchCV for hyperparameter tuning
        svm_grid_search = GridSearchCV(svm_model, hyperparameters, cv=5, scoring='r2', verbose=1, n_jobs=-1)
        svm_grid_search.fit(data_train, labels_train)

        # Getting the best trained model
        best_svm_model = svm_grid_search.best_estimator_

        # Making predictions on test data
        svm_predictions = best_svm_model.predict(data_test)

        # Metrics Calculations
        svm_mae = mean_absolute_error(labels_test, svm_predictions)
        svm_mse = mean_squared_error(labels_test, svm_predictions)
        svm_rmse = np.sqrt(svm_mse)
        svm_mape = np.mean(np.abs((labels_test - svm_predictions) / labels_test)) * 100

        # Getting the best parameters and R^2 score
        print("Best parameters found:", svm_grid_search.best_params_)
        print("Best R^2 score found:", svm_grid_search.best_score_)
        print("Mean Absolute Error (MAE):", svm_mae)
        print("Mean Squared Error (MSE):", svm_mse)
        print("Root Mean Squared Error (RMSE):", svm_rmse)
        print("Mean Absolute Percentage Error (MAPE):", svm_mape)
        print("-----------------------------------------------------\n")

        # Plotting predictions and residuals
        models_predictions_plot(labels_test, svm_predictions, 'SVM')
        models_residuals_plot(labels_test, svm_predictions, 'SVM')

        return best_svm_model