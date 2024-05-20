import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, make_scorer, mean_absolute_error
from sklearn.model_selection import GridSearchCV

# Importing custom functions from shared_functions module
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot

class LinearRegressionModel:
    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        """
        Fits and evaluates a Linear Regression model.

        Parameters:
        - data_train: Training data
        - labels_train: Labels for training data
        - data_test: Test data
        - labels_test: Labels for test data

        Returns:
        - best_linear_regression_model: Best trained Linear Regression model
        """

        # Linear Regression Model
        print("Linear Regression Model fit and evaluation...")

        # Initializing Linear Regression model
        linear_model = LinearRegression()

        # Hyperparameters grid for GridSearchCV
        param_grid = {
            'fit_intercept': [True, False],
            'positive': [True, False],
            'copy_X': [True, False]
        }

        # Scoring methods for GridSearchCV
        scoring = {'mse': make_scorer(mean_squared_error),
                   'r2': make_scorer(r2_score)}

        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(linear_model, param_grid, scoring=scoring, cv=10, refit='r2',
                                   return_train_score=True)

        # Fitting the model and tuning hyperparameters
        grid_search.fit(data_train, labels_train)

        # Getting the best hyperparameters
        best_params = grid_search.best_params_
        print("Best Hyperparameters:", best_params)

        # Getting the best trained model
        best_linear_regression_model = grid_search.best_estimator_

        # Making predictions on test data
        linear_predictions = best_linear_regression_model.predict(data_test)

        # Calculating Metrics
        linear_r2 = r2_score(labels_test, linear_predictions)
        linear_mae = mean_absolute_error(labels_test, linear_predictions)
        linear_mse = mean_squared_error(labels_test, linear_predictions)
        linear_rmse = np.sqrt(linear_mse)
        linear_mape = np.mean(np.abs((labels_test - linear_predictions) / labels_test)) * 100

        # Printing evaluation metrics
        print("Linear Regression R^2:", linear_r2)
        print("Mean Absolute Error (MAE):", linear_mae)
        print("Mean Squared Error (MSE):", linear_mse)
        print("Root Mean Squared Error (RMSE):", linear_rmse)
        print("Mean Absolute Percentage Error (MAPE):", linear_mape)
        print("-----------------------------------------------------\n")

        # Plotting predictions and residuals
        models_predictions_plot(labels_test, linear_predictions, 'Linear Regression')
        models_residuals_plot(labels_test, linear_predictions, 'Linear Regression')

        return best_linear_regression_model