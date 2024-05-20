import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from Project.Classes.Shared.visualize_data import VisualizeData

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
    print(f"{value_type}: {value}\n")


def data_for_KNN(data):
    data_encoded_train = pd.get_dummies(data.data_train['location'], dtype=int)
    data_encoded_test = pd.get_dummies(data.data_test['location'], dtype=int)

    data_encoded_train.fillna(0, inplace=True)
    data_encoded_test.fillna(0, inplace=True)

    data_encoded_test = data_encoded_test.reindex(columns=data_encoded_train.columns, fill_value=0)

    data_train_with_dummies = pd.concat([data.data_train.drop('location', axis=1), data_encoded_train], axis=1)
    data_test_with_dummies = pd.concat([data.data_test.drop('location', axis=1), data_encoded_test], axis=1)

    X_train_data = data_train_with_dummies.drop(['price', 'price_per_sqft'], axis=1)
    X_test_data = data_test_with_dummies.drop(['price', 'price_per_sqft'], axis=1)
    y_train_data = data.labels_train
    y_test_data = data.labels_test

    X_train_data.reset_index(drop=True, inplace=True)
    y_train_data.reset_index(drop=True, inplace=True)

    return X_train_data, X_test_data, y_train_data, y_test_data

def normalize_data_for_clustering(data_train, data_test):
    X_train = data_train.iloc[:, :14]
    X_test = data_test.iloc[:, :14]

    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        X_train = X_train[~np.isnan(X_train).any(axis=1)]
        X_train = X_train[~np.isinf(X_train).any(axis=1)]

    if np.any(np.isnan(X_test)) or np.any(np.isinf(X_test)):
        X_test = X_test[~np.isnan(X_test).any(axis=1)]
        X_test = X_test[~np.isinf(X_test).any(axis=1)]

    scaler = StandardScaler()
    data_train_normalized = scaler.fit_transform(X_train)
    data_test_normalized = scaler.transform(X_test)

    data_train_normalized = pd.DataFrame(data_train_normalized, columns=X_train.columns)
    data_test_normalized = pd.DataFrame(data_test_normalized, columns=X_test.columns)

    return data_train_normalized, data_test_normalized

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