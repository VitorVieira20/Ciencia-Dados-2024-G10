import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy.stats import ttest_ind, f_oneway, ttest_rel
from sklearn.decomposition import PCA
import umap
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, train_test_split

class CleanData:
    '''
    A class to clean and pre-processing the data
    '''

    def __init__(self, data):
        '''
        Initializes the CleaData object

        :param data: The input dataset
        '''
        self.data = data

    def clean_data(self):
        '''
        Cleans the data
        :return: Cleaned Dataset
        '''

        # Read the file passed in constructor
        df = pd.read_csv(self.data)

        # Drop unnecessary columns
        df_drop_cols = df.drop(['area_type','society','availability'],axis='columns')

        # Drop null values
        df_drop_null = df_drop_cols.dropna()

        # In the column siz there are two types of size
        # BHK and Bedroom, so were going to add a column with those data simplified
        df_drop_null['rooms'] = df_drop_null['size'].apply(lambda x: int(x.split(' ')[0]))

        # Drop size column because we have now the bedroom column
        df_drop_size = df_drop_null.drop(['size'], axis='columns')

        '''
        In the column total_sqt are many types of values
            ->Single values: 2400
            ->Range values: 2100 - 2600
            ->Values in Sqr Meter: 34.46Sq. Meter
            ->Values in Sqr Yards: 151.11Sq. Yards
        So were gonna make all of them in a single value like the first example
        '''
        df_conv_val = df_drop_size.copy()
        df_conv_val['total_sqft'] = df_drop_size['total_sqft'].apply(self._convert_values)

        # Remove NaN values
        df_conv_val_cleaned = df_conv_val.dropna()

        '''
        The following steps is for Dimensional Reduction
        We're gonna drop the locations with less than 10 rows of data
        '''

        # Order by "location" column
        df_conv_val_cleaned_sorted = df_conv_val_cleaned.sort_values(by='location')

        # Group by 'location' and count rows number
        location_counts = df_conv_val_cleaned_sorted.groupby('location').size()

        # Filter by location with more or equal 10 rows
        valid_locations = location_counts[location_counts >= 10].index.tolist()

        # Filter DataFrame with valid locations
        df_filtered = df_conv_val_cleaned_sorted[df_conv_val_cleaned_sorted['location'].isin(valid_locations)]

        df_filtered['price_per_sqft'] = df_filtered['price'] / df_filtered['total_sqft']

        '''
        Many business managers will tell you that average sqft per bedroom is 300
        With this, we will remove some outliers
        '''
        df_remove_first_outliers = df_filtered[~(df_filtered.total_sqft / df_filtered.rooms < 300)]

        # Return the cleaned DataFrame
        return df_remove_first_outliers

    def _convert_values(self, x):
        '''
        Calculates the average of a given range like 2100 - 2500
        Converts Sqr. Meter to Sqr. Feat
        Converts Sqr. Yard to Sqr. Feat
        :param x:
        :return:
        '''
        tokens = x.split('-')
        if len(tokens) == 2:
            return (float(tokens[0]) + float(tokens[1])) / 2
        else:
            # 142.84Sq. Meter
            # 300Sq. Yards
            if 'Sq. Meter' in x:
                # Split the Sq. Meter text from the number
                numeric_part = x.split('Sq. Meter')[0].strip()
                numeric_value = ''.join(filter(lambda char: char.isdigit() or char == '.', numeric_part))
                if numeric_value:
                    sqm_value = float(numeric_value)
                    sqft_value = sqm_value * 10.7639
                    return sqft_value
            elif 'Sq. Yards' in x:
                # Split the Sq. Yards text from the number
                numeric_part = x.split('Sq. Yards')[0].strip()
                numeric_value = ''.join(filter(lambda char: char.isdigit() or char == '.', numeric_part))
                if numeric_value:
                    sqy_value = float(numeric_value)
                    sqft_value = sqy_value * 9
                    return sqft_value
            else:
                # Normal values
                try:
                    return pd.to_numeric(x)
                # Other values, like Acres, Perch, etc.
                except ValueError:
                    return None

class RemoveOutliers:
    '''
    A class to remove outliers of the DataFrame
    '''
    def remove_prices_outliers(self, df):
        '''
        Remove outliers per location using mean and one standard deviation
        :param df: DataFrame to remove the outliers
        :return: Cleaned DataFrame
        '''
        df_out = pd.DataFrame()
        for key, subdf in df.groupby('location'):
            m = np.mean(subdf.price_per_sqft)
            st = np.std(subdf.price_per_sqft)
            reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
            df_out = pd.concat([df_out, reduced_df], ignore_index=True)
        return df_out

    def remove_bedroom_outliers(self, df):
        '''
        Removes more outliers based on the price and the bedrooms number
        We remove the data where 2 bedrooms apartments whose price_per_sqft is less than mean of 1 bedroom apartment
        :param df: DataFrame to remove the outliers
        :return: Cleaned DataFrame
        '''
        exclude_indices = np.array([])
        for location, location_df in df.groupby('location'):
            room_stats = {}
            for room, room_df in location_df.groupby('rooms'):
                room_stats[room] = {
                    'mean': np.mean(room_df.price_per_sqft),
                    'std': np.std(room_df.price_per_sqft),
                    'count': room_df.shape[0]
                }
            for room, room_df in location_df.groupby('rooms'):
                stats = room_stats.get(room - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices,
                                                room_df[room_df.price_per_sqft < (stats['mean'])].index.values)
        return df.drop(exclude_indices, axis='index')


class VisualizeData:
    '''
    A class to visualize the data of the DataFrame
    '''

    def plot_scatter_chart(self, df, location):
        '''
        Shows the scatter plot of a certain locations based on their 2 and 3 bedrooms data
        :param df: DataFrame containing all the data
        :param location: Specific location in the DataFrame
        '''
        room2 = df[(df['location'] == location) & (df['rooms'] == 2)]
        room3 = df[(df['location'] == location) & (df['rooms'] == 3)]
        matplotlib.rcParams['figure.figsize'] = (15, 10)
        plt.scatter(room2['total_sqft'], room2['price'], color='blue', label='2 Rooms', s=50)
        plt.scatter(room3['total_sqft'], room3['price'], marker='p', color='red', label='3 Rooms', s=50)
        plt.xlabel("Total Square Feet Area")
        plt.ylabel("Price (Lakh Indian Rupees)")
        plt.title(location)
        plt.legend()
        plt.show()

    def plot_boxplot(self, df, location):
        '''
        Shows the boxplot of a certain location based on their 2 and 3 bedrooms data
        :param df: DataFrame containing all the data
        :param location: Specific location in the DataFrame
        '''
        room2 = df[(df['location'] == location) & (df['rooms'] == 2)]
        room3 = df[(df['location'] == location) & (df['rooms'] == 3)]
        plt.figure(figsize=(15, 10))
        sns.boxplot(x='rooms', y='price', data=pd.concat([room2, room3]), hue='rooms')
        plt.xlabel("Number of Rooms")
        plt.ylabel("Price (Lakh Indian Rupees)")
        plt.title(f'Boxplot for {location}')
        plt.legend(title='Number of Rooms')
        plt.show()

    def plot_violinplot(self, df, location):
        '''
        Shows the violinplot of a certain location based on their 2 and 3 bedrooms data
        :param df: DataFrame containing all the data
        :param location: Specific location in the DataFrame
        '''
        room2 = df[(df['location'] == location) & (df['rooms'] == 2)]
        room3 = df[(df['location'] == location) & (df['rooms'] == 3)]
        plt.figure(figsize=(15, 10))
        sns.violinplot(x='rooms', y='price', data=pd.concat([room2, room3]), hue='rooms')
        plt.xlabel("Number of Rooms")
        plt.ylabel("Price (Lakh Indian Rupees)")
        plt.title(f'Violinplot for {location}')
        plt.legend(title='Number of Rooms')
        plt.show()

    def plot_histogram(self, df, column, location):
        '''
        Shows the histogram of a certain column in the DataFrame for a specific location
        :param df: DataFrame containing all the data
        :param column: Specific column in the DataFrame
        :param location: Specific location in the DataFrame
        '''
        plt.figure(figsize=(8, 6))
        sns.histplot(df[df['location'] == location][column], kde=True)
        plt.xlabel(column)
        plt.ylabel("Frequency")
        plt.title(f'Histogram of {column} in {location}')
        plt.show()

class DimensionalityReduction:
    def __init__(self, data, targets):
        """
        Initialize the DimensionalityReduction object with the dataset.

        Parameters:
        - data: The dataset to perform dimensionality reduction on.
        - targets: The targets of the samples.
        """
        self.data = StandardScaler().fit_transform(data)
        self.targets = targets

    def compute_pca(self, n_components=2):
        """
        Compute Principal Component Analysis (PCA) on the dataset.

        Parameters:
        - n_components: The number of components to keep.

        Returns:
        - pca_projection: The projected data using PCA.
        """
        return PCA(n_components=n_components).fit_transform(self.data)

    def compute_umap(self, n_components=2, n_neighbors=8, min_dist=0.5, metric='euclidean'):
        """
        Compute Uniform Manifold Approximation and Projection (UMAP) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - n_neighbors: The number of neighbors to consider for each point.
        - min_dist: The minimum distance between embedded points.
        - metric: The distance metric to use.

        Returns:
        - umap_projection: The projected data using UMAP.
        """
        return umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist,
                         metric=metric).fit_transform(self.data)

    def plot_projection(self, projection, title):
        """
        Plot the 2D projection of the dataset.

        Parameters:
        - projection: The projected data.
        - title: The title of the plot.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(projection[:, 0], projection[:, 1], c=self.targets, alpha=0.5)
        plt.title(title)
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.show()

class HypothesisTester:
    """
    The t-test assumes that the data is normally distributed and that the variances are equal between groups (for
    unpaired t-test) or within groups (for paired t-test).
    The ANOVA test assumes that the data is normally distributed and that the variances are equal between groups.
    """
    def unpaired_t_test(self, group1, group2):
        """
        Perform unpaired t-test for two groups.

        Parameters:
        - group1: List or array-like object containing data for group 1.
        - group2: List or array-like object containing data for group 2.

        Returns:
        - t_statistic: The calculated t-statistic.
        - p_value: The p-value associated with the t-statistic.
        """
        t_statistic, p_value = ttest_ind(group1, group2)
        return t_statistic, p_value

    def unpaired_anova(self, *groups):
        """
        Perform unpaired ANOVA for more than two groups.

        Parameters:
        - *groups: Variable length argument containing data for each group. Each argument should be a list or array-like
        object.

        Returns:
        - f_statistic: The calculated F-statistic.
        - p_value: The p-value associated with the F-statistic.
        """
        f_statistic, p_value = f_oneway(*groups)
        return f_statistic, p_value

    def paired_t_test(self, group1, group2):
        """
        Perform paired t-test for two groups.

        Parameters:
        - group1: List or array-like object containing data for group 1.
        - group2: List or array-like object containing data for group 2.
                  Should have the same length as group1.

        Returns:
        - t_statistic: The calculated t-statistic.
        - p_value: The p-value associated with the t-statistic.
        """
        t_statistic, p_value = ttest_rel(group1, group2)
        return t_statistic, p_value

class ModelSelection:
    """
    A class to perform model selection for regression problems.
    """
    def __init__(self, X, y):
        """
        Initializes the ModelSelection object with the features and target variable.

        :param X: The input features.
        :param y: The target variable.
        """
        self.X = X
        self.y = y

    def select_model(self):
        """
        Performs model selection by training various models and evaluating them using Mean Squared Error.
        The models used are: Linear Regression, Decision Tree Regression, Random Forest Regression,
        Gradient Boosting Regression, and Support Vector Regression.
        """
        # Create a list of models to evaluate
        models = [LinearRegression(), DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(),
                  SVR(), Ridge()]

        for model in models:
            # Perform K-Fold Cross Validation
            scores = cross_val_score(model, self.X, self.y, cv=10, scoring='neg_mean_squared_error')

            # Calculate Mean Squared Error
            mse_scores = -scores

            # Print the model name and its Mean Squared Error
            print(f'Model: {model.__class__.__name__}, Mean Squared Error: {mse_scores.mean()}')

            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

            # Fit the model to the training data
            model.fit(X_train, y_train)

            # Initialize the VisualizeData object
            vd = VisualizeData()

            # Plot the model's predictions
            vd.plot_model_predictions(X_test, y_test, model)

if __name__ == '__main__':
    # Initialize CleanData object
    cd = CleanData('bengaluru_house_prices.csv')

    # Capture the cleaned DataFrame
    cleaned_data = cd.clean_data()

    # Initialize the RemoveOutliers object
    rm = RemoveOutliers()

    # Initialize the VisualizeData object
    vd = VisualizeData()

    # Show data before outliers in scatter chart
    #vd.plot_scatter_chart(cleaned_data, "Rajaji Nagar")
    #vd.plot_scatter_chart(cleaned_data, "Hebbal")
    #vd.plot_scatter_chart(cleaned_data, "Yeshwanthpur")

    # Show data before outliers in box plots
    #vd.plot_boxplot(cleaned_data, "Rajaji Nagar")
    #vd.plot_boxplot(cleaned_data, "Hebbal")
    #vd.plot_boxplot(cleaned_data, "Yeshwanthpur")

    # Show data before outliers in violins plot
    #vd.plot_violinplot(cleaned_data, "Rajaji Nagar")
    #vd.plot_violinplot(cleaned_data, "Hebbal")
    #vd.plot_violinplot(cleaned_data, "Yeshwanthpur")

    # Show data before outliers in histograms
    #vd.plot_histogram(cleaned_data, "price_per_sqft", "Rajaji Nagar")
    #vd.plot_histogram(cleaned_data, "price_per_sqft", "Hebbal")
    #vd.plot_histogram(cleaned_data, "price_per_sqft", "Yeshwanthpur")

    # Removes first demand of outliers
    first_outliers = rm.remove_prices_outliers(cleaned_data)

    # See the scatter plots before the second removal
    #vd.plot_scatter_chart(first_outliers, "Rajaji Nagar")
    #vd.plot_scatter_chart(first_outliers, "Hebbal")
    #vd.plot_scatter_chart(first_outliers, "Yeshwanthpur")

    # Show data before the second removal in box plots
    #vd.plot_boxplot(first_outliers, "Rajaji Nagar")
    #vd.plot_boxplot(first_outliers, "Hebbal")
    #vd.plot_boxplot(first_outliers, "Yeshwanthpur")

    # Show data before the second removals in violins plot
    #vd.plot_violinplot(first_outliers, "Rajaji Nagar")
    #vd.plot_violinplot(first_outliers, "Hebbal")
    #vd.plot_violinplot(first_outliers, "Yeshwanthpur")

    # Show data before the second removal in histograms
    #vd.plot_histogram(first_outliers, "price_per_sqft", "Rajaji Nagar")
    #vd.plot_histogram(first_outliers, "price_per_sqft", "Hebbal")
    #vd.plot_histogram(first_outliers, "price_per_sqft", "Yeshwanthpur")

    # Removes second demand of outliers
    second_outliers = rm.remove_bedroom_outliers(first_outliers)

    # See the scatter plots after the second removal
    #vd.plot_scatter_chart(second_outliers, "Rajaji Nagar")
    #vd.plot_scatter_chart(second_outliers, "Hebbal")
    #vd.plot_scatter_chart(second_outliers, "Yeshwanthpur")

    # Show data after the second removal in box plots
    #vd.plot_boxplot(second_outliers, "Rajaji Nagar")
    #vd.plot_boxplot(second_outliers, "Hebbal")
    #vd.plot_boxplot(second_outliers, "Yeshwanthpur")

    # Show data after the second removal in violins plot
    vd.plot_violinplot(second_outliers, "Rajaji Nagar")
    vd.plot_violinplot(second_outliers, "Hebbal")
    vd.plot_violinplot(second_outliers, "Yeshwanthpur")

    # Show data after the second removal in histograms
    vd.plot_histogram(second_outliers, "price_per_sqft", "Rajaji Nagar")
    vd.plot_histogram(second_outliers, "price_per_sqft", "Hebbal")
    vd.plot_histogram(second_outliers, "price_per_sqft", "Yeshwanthpur")

    data = second_outliers[['total_sqft', 'bath', 'balcony', 'rooms', 'price_per_sqft']]
    targets = second_outliers['price']

    # Initialize DimensionalityReduction object
    dr = DimensionalityReduction(np.array(data), targets)

    dr.plot_projection(dr.compute_pca(), 'PCA Projection')
    dr.plot_projection(dr.compute_umap(), 'UMAP Projection')

    # Prices per SQq. feet for three locations
    price_sqft_Rajaji_Nagar = second_outliers[second_outliers['location'] == 'Rajaji Nagar']['price_per_sqft']
    price_sqft_Hebbal = second_outliers[second_outliers['location'] == 'Hebbal']['price_per_sqft']
    price_sqft_Yeshwanthpur = second_outliers[second_outliers['location'] == 'Yeshwanthpur']['price_per_sqft']

    # Prices for Rajaji Nagar location
    price_Rajaji_Nagar = second_outliers[second_outliers['location'] == 'Rajaji Nagar']['price']

    # Total Sq. Feet for Rajaji Nagar location
    total_sqft_Rajaji_Nagar = second_outliers[second_outliers['location'] == 'Rajaji Nagar']['total_sqft']

    # Initialize the HypothesisTester class with the data
    tester = HypothesisTester()

    # Perform unpaired t-test between Hebbal and Rajaji Nagar locations
    t_stat, p_val = tester.unpaired_t_test(price_sqft_Hebbal, price_sqft_Rajaji_Nagar)
    print("\nUnpaired t-test between Hebbal and Rajaji Nagar locations:")
    print("t-statistic:", t_stat)
    print("p-value:", p_val)

    # Perform unpaired ANOVA among the three locations
    f_stat, p_val_anova = tester.unpaired_anova(price_sqft_Rajaji_Nagar, price_sqft_Hebbal, price_sqft_Yeshwanthpur)
    print("\nUnpaired ANOVA among three locations:")
    print("F-statistic:", f_stat)
    print("p-value:", p_val_anova)

    # Perform paired t-test for total_sqft and price within Rajaji Nagar location
    t_stat, p_val = tester.paired_t_test(total_sqft_Rajaji_Nagar, price_Rajaji_Nagar)
    print("\nPaired t-test for total_sqft and price within Rajaji Nagar location:")
    print("t-statistic:", t_stat)
    print("p-value:", p_val)

    # 10 new features

    # Price distribution per bedroom.
    second_outliers['price_per_bedroom'] = second_outliers['price'] / second_outliers['rooms']

    # Bathroom to bedroom ratio.
    second_outliers['bath_per_bedroom'] = second_outliers['bath'] / second_outliers['rooms']

    # Space distribution per bedroom.
    second_outliers['total_sqft_per_room'] = second_outliers['total_sqft'] / second_outliers['rooms']

    # Balcony to bedroom ratio
    second_outliers['balcony_per_room'] = second_outliers['balcony'] / second_outliers['rooms']

    # Price distribution per bathroom
    second_outliers['price_per_bath'] = second_outliers['price'] / second_outliers['bath']

    # Space distribution per bathroom
    second_outliers['total_sqft_per_bath'] = second_outliers['total_sqft'] / second_outliers['bath']

    # Balcony to bathroom ratio
    second_outliers['balcony_per_bath'] = second_outliers['balcony'] / second_outliers['bath']

    # Price distribution per balcony
    second_outliers['price_per_balcony'] = second_outliers['price'] / second_outliers['balcony']

    # Space distribution per balcony
    second_outliers['total_sqft_per_balcony'] = second_outliers['total_sqft'] / second_outliers['balcony']

    # Bedroom to bathroom ratio
    second_outliers['room_per_bath'] = second_outliers['rooms'] / second_outliers['bath']

    # Model selection
    second_outliers_encoded = pd.get_dummies(second_outliers)
    second_outliers_encoded.fillna(second_outliers_encoded.mean(), inplace=True)

    for column in second_outliers_encoded.columns:
        if np.isinf(second_outliers_encoded[column]).any():
            max_val = second_outliers_encoded[~np.isinf(second_outliers_encoded[column])][column].max()
            second_outliers_encoded[column] = second_outliers_encoded[column].replace([np.inf, -np.inf], max_val)

    # Now, 'second_outliers_encoded' is your DataFrame with encoded categorical variables
    X = second_outliers_encoded.drop(['price'], axis=1)
    y = second_outliers_encoded['price']

    # Initialize the ModelSelection class with the data
    ms = ModelSelection(X, y)

    # Perform model selection
    ms.select_model()