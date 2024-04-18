from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot, models_training_loss_plot

class RandomForestModel:
    def _fit_model(self, data_train, labels_train, params):
        rf_model = RandomForestRegressor(**params)
        rf_model.fit(data_train, labels_train)
        return rf_model

    def _evaluate_model(self, model, data_test, labels_test):
        predictions = model.predict(data_test)
        mse = mean_squared_error(labels_test, predictions)
        r2 = r2_score(labels_test, predictions)
        return {'mse': mse, 'r2': r2}

    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        hyperparameters = [
            {'n_estimators': 100, 'random_state': 42},
            {'n_estimators': 200, 'random_state': 42},
            {'n_estimators': 500, 'random_state': 42},
            {'n_estimators': 1000, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 5, 'random_state': 42},
            {'n_estimators': 100, 'max_depth': 10, 'random_state': 42},
            {'n_estimators': 100, 'min_samples_split': 2, 'random_state': 42},
            {'n_estimators': 100, 'min_samples_split': 5, 'random_state': 42},
            {'n_estimators': 100, 'min_samples_leaf': 1, 'random_state': 42},
            {'n_estimators': 100, 'min_samples_leaf': 2, 'random_state': 42},
        ]

        results = []

        for params in hyperparameters:
            rf_model = self._fit_model(data_train, labels_train, params)
            evaluation_result = self._evaluate_model(rf_model, data_test, labels_test)
            results.append({'params': params, 'mse': evaluation_result['mse'], 'r2': evaluation_result['r2']})

        sorted_results = sorted(results, key=lambda x: x['r2'], reverse=True)

        for result in sorted_results:
            print("Parameters:", result['params'])
            print("Random Forest MSE:", result['mse'])
            print("Random Forest R^2:", result['r2'])
            print("---------------------------------------------")

        best_params = max(results, key=lambda x: x['r2'])['params']
        best_model = self._fit_model(data_train, labels_train, best_params)

        print("Best Random Forest Parameters:", best_params)
        evaluation_result = self._evaluate_model(best_model, data_test, labels_test)
        print("Random Forest MSE:", evaluation_result['mse'])
        print("Random Forest R^2:", evaluation_result['r2'])

        models_predictions_plot(labels_test, best_model.predict(data_test), 'Random Forest')
        models_residuals_plot(labels_test, best_model.predict(data_test), 'Random Forest')

        return best_model