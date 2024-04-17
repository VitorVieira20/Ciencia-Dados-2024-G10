import pandas as pd

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

    X_train = pd.concat([data.data_train.drop('location', axis=1), data_train_encoded], axis=1)
    y_train = data.labels_train
    X_test = pd.concat([data.data_test.drop('location', axis=1), data_test_encoded], axis=1)
    y_test = data.labels_test

    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)

    return X_train, X_test, y_train, y_test