from Project.Classes.visualize_data import VisualizeData
from Project.Classes.hypothesis_tester import HypothesisTester

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