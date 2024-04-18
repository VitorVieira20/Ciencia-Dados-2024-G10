from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot, models_training_loss_plot

class GradientBoostingModel:
    def _fit_model(self, data_train, labels_train, params):
        gb_model = GradientBoostingRegressor(**params)
        gb_model.fit(data_train, labels_train)
        return gb_model

    def _evaluate_model(self, model, data_test, labels_test):
        gb_predictions = model.predict(data_test)
        mse = mean_squared_error(labels_test, gb_predictions)
        r2 = r2_score(labels_test, gb_predictions)
        return {'mse': mse, 'r2': r2}

    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        hyperparameters = [
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42},
            {'n_estimators': 200, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 42},
            {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 3, 'random_state': 42},
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5, 'random_state': 42},
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 3, 'random_state': 0},
            {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 5, 'random_state': 0},
            {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 3, 'random_state': 42},
            {'n_estimators': 150, 'learning_rate': 0.1, 'max_depth': 4, 'random_state': 42},
            {'n_estimators': 100, 'learning_rate': 0.01, 'max_depth': 3, 'random_state': 42},
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6, 'random_state': 42},
        ]

        results = []

        for params in hyperparameters:
            gb_model = self._fit_model(data_train, labels_train, params)
            evaluation_result = self._evaluate_model(gb_model, data_test, labels_test)
            results.append({'params': params, 'mse': evaluation_result['mse'], 'r2': evaluation_result['r2']})

        sorted_results = sorted(results, key=lambda x: x['r2'], reverse=True)

        for result in sorted_results:
            print("Parameters:", result['params'])
            print("Gradient Boosting MSE:", result['mse'])
            print("Gradient Boosting R^2:", result['r2'])
            print("---------------------------------------------")

        best_params = max(results, key=lambda x: x['r2'])['params']
        best_model = self._fit_model(data_train, labels_train, best_params)

        print("Best Gradient Boosting Parameters:", best_params)
        evaluation_result = self._evaluate_model(best_model, data_test, labels_test)
        print("Gradient Boosting MSE:", evaluation_result['mse'])
        print("Gradient Boosting R^2:", evaluation_result['r2'])

        models_predictions_plot(labels_test, best_model.predict(data_test), 'Gradient Boosting')
        models_residuals_plot(labels_test, best_model.predict(data_test), 'Gradient Boosting')

        return best_model
