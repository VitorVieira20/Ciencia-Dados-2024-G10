from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from Project.Classes.Shared.shared_functions import models_predictions_plot, models_residuals_plot, models_training_loss_plot

class LinearRegressionModel:
    def fit_and_evaluate(self, data_train, labels_train, data_test, labels_test):
        # Linear Regression Model
        linear_model = LinearRegression()
        linear_model.fit(data_train, labels_train)
        linear_predictions = linear_model.predict(data_test)

        linear_mse = mean_squared_error(labels_test, linear_predictions)
        linear_r2 = r2_score(labels_test, linear_predictions)
        print("Linear Regression MSE:", linear_mse)
        print("Linear Regression R^2:", linear_r2)

        models_predictions_plot(labels_test, linear_predictions, 'Linear Regression')
        models_residuals_plot(labels_test, linear_predictions, 'Linear Regression')

        return linear_model