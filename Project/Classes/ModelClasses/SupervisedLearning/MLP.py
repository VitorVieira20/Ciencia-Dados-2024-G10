from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot, models_training_loss_plot

class MLPModel:
    def _fit_model(self, data_train, labels_train, params):
        mlp_model = MLPRegressor(**params)
        mlp_model.fit(data_train, labels_train)
        return mlp_model

    def _evaluate_model(self, model, data_test, labels_test):
        mlp_predictions = model.predict(data_test)
        mse = mean_squared_error(labels_test, mlp_predictions)
        r2 = r2_score(labels_test, mlp_predictions)
        return {'mse': mse, 'r2': r2}

    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        hyperparameters = [
            {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 500},
            {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'solver': 'adam', 'max_iter': 500},
            {'hidden_layer_sizes': (50,), 'activation': 'logistic', 'solver': 'sgd', 'max_iter': 1000},
            {'hidden_layer_sizes': (50, 20), 'activation': 'tanh', 'solver': 'adam', 'max_iter': 1000},
            {'hidden_layer_sizes': (50, 50, 50), 'activation': 'relu', 'solver': 'adam', 'max_iter': 1000},
            {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 1000, 'alpha': 0.0001},
            {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 1000, 'learning_rate': 'constant'},
            {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 1000, 'learning_rate_init': 0.001},
        ]

        results = []

        for params in hyperparameters:
            mlp_model = self._fit_model(data_train, labels_train, params)
            evaluation_result = self._evaluate_model(mlp_model, data_test, labels_test)
            results.append({'params': params, 'mse': evaluation_result['mse'], 'r2': evaluation_result['r2']})

        sorted_results = sorted(results, key=lambda x: x['r2'], reverse=True)

        for result in sorted_results:
            print("Parameters:", result['params'])
            print("MLP MSE:", result['mse'])
            print("MLP R^2:", result['r2'])
            print("---------------------------------------------")

        best_params = max(results, key=lambda x: x['r2'])['params']
        best_model = self._fit_model(data_train, labels_train, best_params)

        print("Best MLP Parameters:", best_params)
        evaluation_result = self._evaluate_model(best_model, data_test, labels_test)
        print("MLP MSE:", evaluation_result['mse'])
        print("MLP R^2:", evaluation_result['r2'])

        models_predictions_plot(labels_test, best_model.predict(data_test), 'MLP')
        models_residuals_plot(labels_test, best_model.predict(data_test), 'MLP')
        models_training_loss_plot(best_model, 'MLP')

        return best_model
