import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Importing custom functions from shared_functions module
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot, models_training_loss_plot

class GradientBoostingModel:
    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        """
        Fits and evaluates a Gradient Boosting model.

        Parameters:
        - data_train: Training data
        - labels_train: Labels for training data
        - data_test: Test data
        - labels_test: Labels for test data

        Returns:
        - best_gb_model: Best trained Gradient Boosting model
        """

        # Gradient Boosting Model
        print("Gradient Boosting Model fit and evaluation...")

        # Hyperparameters grid for GridSearchCV
        hyperparameters = {
            'n_estimators': [100, 150, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5, 6],
            'random_state': [0, 42]
        }

        # Initializing Gradient Boosting model
        gb_model = GradientBoostingRegressor()

        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(gb_model, hyperparameters, scoring='r2', cv=5, n_jobs=-1)

        # Fitting the model and tuning hyperparameters
        grid_search.fit(data_train, labels_train)

        # Getting the best trained model
        best_gb_model = grid_search.best_estimator_

        # Making predictions on test data
        gb_predictions = best_gb_model.predict(data_test)

        # Metrics Calculations
        gb_mae = mean_absolute_error(labels_test, gb_predictions)
        gb_mse = mean_squared_error(labels_test, gb_predictions)
        gb_rmse = np.sqrt(gb_mse)
        gb_mape = np.mean(np.abs((labels_test - gb_predictions) / labels_test)) * 100

        # Getting the best parameters and R^2 score
        print("Best Gradient Boosting Parameters:", grid_search.best_params_)
        print("Best R^2 Score:", grid_search.best_score_)
        print("Mean Absolute Error (MAE):", gb_mae)
        print("Mean Squared Error (MSE):", gb_mse)
        print("Root Mean Squared Error (RMSE):", gb_rmse)
        print("Mean Absolute Percentage Error (MAPE):", gb_mape)
        print("-----------------------------------------------------\n")

        # Plotting predictions and residuals
        models_predictions_plot(labels_test, gb_predictions, 'Gradient Boosting')
        models_residuals_plot(labels_test, gb_predictions, 'Gradient Boosting')

        return best_gb_model