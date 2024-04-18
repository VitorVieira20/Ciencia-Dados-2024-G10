from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot, models_training_loss_plot

class SVMModel:
    def _fit_model(self, data_train, labels_train, params):
        svm_model = SVR(**params)
        svm_model.fit(data_train, labels_train)
        return svm_model

    def _evaluate_model(self, model, data_test, labels_test):
        svm_predictions = model.predict(data_test)
        mse = mean_squared_error(labels_test, svm_predictions)
        r2 = r2_score(labels_test, svm_predictions)
        return {'mse': mse, 'r2': r2}

    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        hyperparameters = [
            {'kernel': 'linear', 'C': 0.1},
            {'kernel': 'linear', 'C': 1},
            {'kernel': 'linear', 'C': 10},
            {'kernel': 'rbf', 'C': 0.1, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 0.1, 'gamma': 'auto'},
            {'kernel': 'rbf', 'C': 1, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 1, 'gamma': 'auto'},
            {'kernel': 'rbf', 'C': 10, 'gamma': 'scale'},
            {'kernel': 'rbf', 'C': 10, 'gamma': 'auto'},
            {'kernel': 'poly', 'C': 0.1, 'degree': 2},
            {'kernel': 'poly', 'C': 0.1, 'degree': 3},
            {'kernel': 'poly', 'C': 1, 'degree': 2},
            {'kernel': 'poly', 'C': 1, 'degree': 3},
            {'kernel': 'poly', 'C': 10, 'degree': 2},
            {'kernel': 'poly', 'C': 10, 'degree': 3},
            {'kernel': 'sigmoid', 'C': 0.1},
            {'kernel': 'sigmoid', 'C': 1},
            {'kernel': 'sigmoid', 'C': 10},
        ]

        results = []

        for params in hyperparameters:
            svm_model = self._fit_model(data_train, labels_train, params)
            evaluation_result = self._evaluate_model(svm_model, data_test, labels_test)
            results.append({'params': params, 'mse': evaluation_result['mse'], 'r2': evaluation_result['r2']})

        sorted_results = sorted(results, key=lambda x: x['r2'], reverse=True)

        for result in sorted_results:
            print("Parameters:", result['params'])
            print("SVM MSE:", result['mse'])
            print("SVM R^2:", result['r2'])
            print("---------------------------------------------")

        best_params = max(results, key=lambda x: x['r2'])['params']
        best_model = self._fit_model(data_train, labels_train, best_params)

        print("Best SVM Parameters:", best_params)
        evaluation_result = self._evaluate_model(best_model, data_test, labels_test)
        print("SVM MSE:", evaluation_result['mse'])
        print("SVM R^2:", evaluation_result['r2'])

        models_predictions_plot(labels_test, best_model.predict(data_test), 'SVM')
        models_residuals_plot(labels_test, best_model.predict(data_test), 'SVM')

        return best_model