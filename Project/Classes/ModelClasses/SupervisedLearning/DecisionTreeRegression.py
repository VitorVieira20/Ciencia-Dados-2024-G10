import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

# Importing custom functions from shared_functions module
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot

class DecisionTreeRegressionModel:
    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        """
        Fits and evaluates a Decision Tree Regressor model.

        Parameters:
        - data_train: Training data
        - labels_train: Labels for training data
        - data_test: Test data
        - labels_test: Labels for test data

        Returns:
        - best_dt_model: Best trained Decision Tree Regressor model
        """

        # Decision Tree Regressor Model
        print("Decision Tree Regressor Model fit and evaluation...")

        # Hyperparameters grid for GridSearchCV
        hyperparameters = {
            'max_depth': [None, 10, 20, 30, 50, 100],
            'min_samples_split': [2, 5, 10, 20, 50],
            'min_samples_leaf': [1, 2, 5, 10, 20],
            'max_features': ['sqrt', 'log2']
        }

        # Initializing Decision Tree Regressor model
        dt_model = DecisionTreeRegressor()

        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(dt_model, hyperparameters, scoring='r2', cv=5, n_jobs=-1)

        # Fitting the model and tuning hyperparameters
        grid_search.fit(data_train, labels_train)

        # Getting the best trained model
        best_dt_model = grid_search.best_estimator_

        # Making predictions on test data
        dt_predictions = best_dt_model.predict(data_test)

        # Calculating metrics
        dt_mae = mean_absolute_error(labels_test, dt_predictions)
        dt_mse = mean_squared_error(labels_test, dt_predictions)
        dt_rmse = np.sqrt(dt_mse)
        dt_mape = np.mean(np.abs((labels_test - dt_predictions) / labels_test)) * 100

        # Getting the best hyperparameters Metrics
        print("Best Hyperparameters:", grid_search.best_params_)
        print("Best R^2 Score:", grid_search.best_score_)
        print("Mean Absolute Error (MAE):", dt_mae)
        print("Mean Squared Error (MSE):", dt_mse)
        print("Root Mean Squared Error (RMSE):", dt_rmse)
        print("Mean Absolute Percentage Error (MAPE):", dt_mape)
        print("-----------------------------------------------------\n")

        # Plotting predictions and residuals
        models_predictions_plot(labels_test, dt_predictions, 'Decision Tree Regression')
        models_residuals_plot(labels_test, dt_predictions, 'Decision Tree Regression')

        return best_dt_model