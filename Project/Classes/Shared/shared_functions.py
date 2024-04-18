import pandas as pd
import matplotlib.pyplot as plt

from Project.Classes.visualize_data import VisualizeData

def print_shape(data, data_name):
    print(f"Data: {data_name}")
    print("Training data shape:", data.data_train.shape)
    print("Training labels shape:", data.labels_train.shape)
    print("Testing data shape:", data.data_test.shape)
    print("Testing labels shape:", data.labels_test.shape)
    print("\n")

def plot_data_visualizations(data, locations):
    vd = VisualizeData()
    for location in locations:
        # Scatter chart
        vd.plot_scatter_chart(data, location)

        # Box plot
        vd.plot_boxplot(data, location)

        # Violin plot
        vd.plot_violinplot(data, location)

        # Histogram
        vd.plot_histogram(data, "price_per_sqft", location)

    # Heatmap
    vd.plot_heatmap(data)

def print_hypothesis_result(title, stat_type, stat, value_type, value):
    print(f"\n{title}")
    print(f"{stat_type}: {stat}")
    print(f"{value_type}: {value}")


def data_for_KNN(data):
    data_train_encoded = pd.get_dummies(data.data_train['location'], dtype=int)
    data_test_encoded = pd.get_dummies(data.data_test['location'], dtype=int)

    data_train_encoded.fillna(0, inplace=True)
    data_test_encoded.fillna(0, inplace=True)

    data_test_encoded = data_test_encoded.reindex(columns=data_train_encoded.columns, fill_value=0)

    X_train_dummies = pd.concat([data.data_train.drop('location', axis=1), data_train_encoded], axis=1)
    X_test_dummies = pd.concat([data.data_test.drop('location', axis=1), data_test_encoded], axis=1)

    X_train = X_train_dummies.drop('price', axis=1)
    X_test = X_test_dummies.drop('price', axis=1)
    y_train = data.labels_train
    y_test = data.labels_test

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test

def models_predictions_plot(test_labels, model_predictions, model_name):
    plt.figure(figsize=(10, 6))
    plt.scatter(test_labels, model_predictions, color='blue', alpha=0.5)
    plt.plot(test_labels, test_labels, color='red')
    plt.title(f'Comparação entre Valores Reais e Previsões ({model_name})')
    plt.xlabel('Valores Reais')
    plt.ylabel(f'Previsões do {model_name}')
    plt.grid(True)
    plt.show()


def models_residuals_plot(test_labels, model_predictions, model_name):
    residuals = test_labels - model_predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(model_predictions, residuals, color='green', alpha=0.5)
    plt.title(f'Residuals vs Predicted Values ({model_name})')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.grid(True)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.show()


def models_training_loss_plot(model, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(model.loss_curve_, label='Training Loss')
    plt.title(f'Training Loss Curve ({model_name})')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()