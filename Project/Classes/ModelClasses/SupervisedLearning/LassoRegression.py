from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot, models_training_loss_plot

class LassoRegressionModel:
    def _fit_model(self, data_train, labels_train, params):
        lasso_model = Lasso(max_iter=10000, **params)
        lasso_model.fit(data_train, labels_train)
        return lasso_model

    def _evaluate_model(self, model, data_test, labels_test):
        lasso_predictions = model.predict(data_test)
        mse = mean_squared_error(labels_test, lasso_predictions)
        r2 = r2_score(labels_test, lasso_predictions)
        return {'mse': mse, 'r2': r2}

    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        hyperparameters = [
            {'alpha': 0.1},
            {'alpha': 0.01},
            {'alpha': 0.001},
            {'alpha': 0.0001},
        ]

        results = []

        for params in hyperparameters:
            lasso_model = self._fit_model(data_train, labels_train, params)
            evaluation_result = self._evaluate_model(lasso_model, data_test, labels_test)
            results.append({'params': params, 'mse': evaluation_result['mse'], 'r2': evaluation_result['r2']})

        sorted_results = sorted(results, key=lambda x: x['r2'], reverse=True)

        for result in sorted_results:
            print("Parameters:", result['params'])
            print("Lasso Regression MSE:", result['mse'])
            print("Lasso Regression R^2:", result['r2'])
            print("---------------------------------------------")

        best_params = max(results, key=lambda x: x['r2'])['params']
        best_model = self._fit_model(data_train, labels_train, best_params)

        print("Best Lasso Regression Parameters:", best_params)
        evaluation_result = self._evaluate_model(best_model, data_test, labels_test)
        print("Lasso Regression MSE:", evaluation_result['mse'])
        print("Lasso Regression R^2:", evaluation_result['r2'])

        models_predictions_plot(labels_test, best_model.predict(data_test), 'Lasso Regression')
        models_residuals_plot(labels_test, best_model.predict(data_test), 'Lasso Regression')

        return best_model
