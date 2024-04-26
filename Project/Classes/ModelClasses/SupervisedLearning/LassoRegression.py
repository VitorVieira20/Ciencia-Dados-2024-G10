from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# Importing custom functions from shared_functions module
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot, models_training_loss_plot

class LassoRegressionModel:
    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        """
        Fits and evaluates a Lasso Regression model.

        Parameters:
        - data_train: Training data
        - labels_train: Labels for training data
        - data_test: Test data
        - labels_test: Labels for test data

        Returns:
        - best_lasso_model: Best trained Lasso Regression model
        """

        # Lasso Regression Model
        print("Lasso Regression Model fit and evaluation...")

        # Hyperparameters grid for GridSearchCV
        hyperparameters = {
            'alpha': [0.1, 0.01, 0.001, 0.0001]
        }

        # Initializing Lasso Regression model
        lasso_model = Lasso(max_iter=10000)

        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(lasso_model, hyperparameters, scoring='r2', cv=5, n_jobs=-1)

        # Fitting the model and tuning hyperparameters
        grid_search.fit(data_train, labels_train)

        # Getting the best parameters and R^2 score
        print("Best Lasso Regression Parameters:", grid_search.best_params_)
        print("Best R^2 Score:", grid_search.best_score_)
        print("-----------------------------------------------------\n")

        # Getting the best trained model
        best_lasso_model = grid_search.best_estimator_

        # Making predictions on test data
        lasso_predictions = best_lasso_model.predict(data_test)

        # Plotting predictions and residuals
        models_predictions_plot(labels_test, lasso_predictions, 'Lasso Regression')
        models_residuals_plot(labels_test, lasso_predictions, 'Lasso Regression')

        return best_lasso_model