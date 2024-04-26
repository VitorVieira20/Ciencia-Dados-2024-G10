from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor

# Importing custom functions from shared_functions module
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot, models_training_loss_plot

class MLPModel:
    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        """
        Fits and evaluates a Multi-layer Perceptron (MLP) model.

        Parameters:
        - data_train: Training data
        - labels_train: Labels for training data
        - data_test: Test data
        - labels_test: Labels for test data

        Returns:
        - best_mlp_model: Best trained MLP model
        """

        # Multi-layer Perceptron (MLP) Model
        print("MLP Model fit and evaluation...")

        # Hyperparameters grid for GridSearchCV
        hyperparameters = {
            'hidden_layer_sizes': [(100,), (100, 50), (50,), (50, 20), (50, 50, 50)],
            'activation': ['relu', 'logistic', 'tanh'],
            'solver': ['adam', 'sgd'],
            'max_iter': [500, 1000],
            'alpha': [0.0001],
            'learning_rate': ['constant'],
            'learning_rate_init': [0.001]
        }

        # Initializing MLP model
        mlp_model = MLPRegressor()

        # GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(mlp_model, hyperparameters, scoring='r2', cv=5, n_jobs=-1)

        # Fitting the model and tuning hyperparameters
        grid_search.fit(data_train, labels_train)

        # Getting the best hyperparameters and R^2 score
        print("Best Hyperparameters:", grid_search.best_params_)
        print("Best R^2 Score:", grid_search.best_score_)
        print("-----------------------------------------------------\n")

        # Getting the best trained model
        best_mlp_model = grid_search.best_estimator_

        # Making predictions on test data
        mlp_predictions = best_mlp_model.predict(data_test)

        # Plotting predictions and residuals
        models_predictions_plot(labels_test, mlp_predictions, 'MLP')
        models_residuals_plot(labels_test, mlp_predictions, 'MLP')

        return best_mlp_model