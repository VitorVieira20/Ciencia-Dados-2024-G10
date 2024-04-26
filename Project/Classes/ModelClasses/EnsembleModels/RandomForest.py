from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Importing custom functions from shared_functions module
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot, models_training_loss_plot

class RandomForestModel:
    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        """
        Fits and evaluates a Random Forest model.

        Parameters:
        - data_train: Training data
        - labels_train: Labels for training data
        - data_test: Test data
        - labels_test: Labels for test data

        Returns:
        - best_rf_model: Best trained Random Forest model
        """

        # Random Forest Model
        print("Random Forest Model fit and evaluation...")

        # Hyperparameters grid for GridSearchCV
        hyperparameters = {
            'n_estimators': [100, 200, 500, 1000],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'random_state': [42]
        }

        # Initializing Random Forest model
        rf_model = RandomForestRegressor()

        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(rf_model, hyperparameters, scoring='r2', cv=5, n_jobs=-1)

        # Fitting the model and tuning hyperparameters
        grid_search.fit(data_train, labels_train)

        # Getting the best parameters and R^2 score
        print("Best Random Forest Parameters:", grid_search.best_params_)
        print("Best R^2 Score:", grid_search.best_score_)

        # Getting the best trained model
        best_rf_model = grid_search.best_estimator_

        # Making predictions on test data
        rf_predictions = best_rf_model.predict(data_test)

        # Plotting predictions and residuals
        models_predictions_plot(labels_test, rf_predictions, 'Random Forest')
        models_residuals_plot(labels_test, rf_predictions, 'Random Forest')

        return best_rf_model