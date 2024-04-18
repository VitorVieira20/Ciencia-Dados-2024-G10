from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot, models_training_loss_plot

class RidgeRegressionModel:
    def _fit_model(self, data_train, labels_train, params):
        ridge_model = Ridge(max_iter=10000, **params)
        ridge_model.fit(data_train, labels_train)
        return ridge_model

    def _evaluate_model(self, model, data_test, labels_test):
        ridge_predictions = model.predict(data_test)
        mse = mean_squared_error(labels_test, ridge_predictions)
        r2 = r2_score(labels_test, ridge_predictions)
        return {'mse': mse, 'r2': r2}

    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        hyperparameters = [
            {'alpha': 0.1},
            {'alpha': 1},
            {'alpha': 10},
            {'alpha': 100},
        ]

        results = []

        for params in hyperparameters:
            ridge_model = self._fit_model(data_train, labels_train, params)
            evaluation_result = self._evaluate_model(ridge_model, data_test, labels_test)
            results.append({'params': params, 'mse': evaluation_result['mse'], 'r2': evaluation_result['r2']})

        sorted_results = sorted(results, key=lambda x: x['r2'], reverse=True)

        for result in sorted_results:
            print("Parameters:", result['params'])
            print("Ridge Regression MSE:", result['mse'])
            print("Ridge Regression R^2:", result['r2'])
            print("---------------------------------------------")

        best_params = max(results, key=lambda x: x['r2'])['params']
        best_model = self._fit_model(data_train, labels_train, best_params)

        print("Best Ridge Regression Parameters:", best_params)
        evaluation_result = self._evaluate_model(best_model, data_test, labels_test)
        print("Ridge Regression MSE:", evaluation_result['mse'])
        print("Ridge Regression R^2:", evaluation_result['r2'])

        models_predictions_plot(labels_test, best_model.predict(data_test), 'Ridge Regression')
        models_residuals_plot(labels_test, best_model.predict(data_test), 'Ridge Regression')

        return best_model
