import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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
        df_drop_null['bedroom'] = df_drop_null['size'].apply(lambda x: int(x.split(' ')[0]))

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
        df_remove_first_outliers = df_filtered[~(df_filtered.total_sqft / df_filtered.bedroom < 300)]

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
            bedroom_stats = {}
            for bedroom, bedroom_df in location_df.groupby('bedroom'):
                bedroom_stats[bedroom] = {
                    'mean': np.mean(bedroom_df.price_per_sqft),
                    'std': np.std(bedroom_df.price_per_sqft),
                    'count': bedroom_df.shape[0]
                }
            for bedroom, bedroom_df in location_df.groupby('bedroom'):
                stats = bedroom_stats.get(bedroom - 1)
                if stats and stats['count'] > 5:
                    exclude_indices = np.append(exclude_indices,
                                                bedroom_df[bedroom_df.price_per_sqft < (stats['mean'])].index.values)
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
        bed2 = df[(df['location'] == location) & (df['bedroom'] == 2)]
        bed3 = df[(df['location'] == location) & (df['bedroom'] == 3)]
        matplotlib.rcParams['figure.figsize'] = (15, 10)
        plt.scatter(bed2['total_sqft'], bed2['price'], color='blue', label='2 Bedroom', s=50)
        plt.scatter(bed3['total_sqft'], bed3['price'], marker='p', color='red', label='3 Bedroom', s=50)
        plt.xlabel("Total Square Feet Area")
        plt.ylabel("Price (Lakh Indian Rupees)")
        plt.title(location)
        plt.legend()
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

    def compute_lda(self, n_components=2):
        """
        Perform Linear Discriminant Analysis (LDA) on the input data.

        Parameters:
        - n_components: The number of components to keep

        Returns:
            array-like: The reduced-dimensional representation of the data using LDA.
        """
        return LinearDiscriminantAnalysis(n_components=n_components).fit_transform(self.data, self.targets)

    def compute_tsne(self, n_components=2, perplexity=3):
        """
        Compute t-Distributed Stochastic Neighbor Embedding (t-SNE) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - perplexity: The perplexity parameter for t-SNE.

        Returns:
        - tsne_projection: The projected data using t-SNE.
        """
        return TSNE(n_components=n_components, perplexity=perplexity).fit_transform(self.data)

    def compute_lle(self, n_components=2, n_neighbors=20):
        """
        Compute Locally Linear Embedding (LLE) on the dataset.

        Parameters:
        - n_components: The number of components to embed the data into.
        - n_neighbors: The number of neighbors to consider for each point.

        Returns:
        - lle_projection: The projected data using LLE.
        """
        return LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components).fit_transform(self.data)

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

if __name__ == '__main__':
    # Initialize CleanData object
    cd = CleanData('bengaluru_house_prices.csv')

    # Capture the cleaned DataFrame
    cleaned_data = cd.clean_data()

    # Initialize the RemoveOutliers object
    rm = RemoveOutliers()

    # Initialize the VisualizeData object
    vd = VisualizeData()

    # Removes first demand of outliers
    first_outliers = rm.remove_prices_outliers(cleaned_data)

    # See the scatter plots before the second removal
    vd.plot_scatter_chart(first_outliers, "Rajaji Nagar")
    vd.plot_scatter_chart(first_outliers, "Hebbal")
    vd.plot_scatter_chart(first_outliers, "Yeshwanthpur")

    # Removes second demand of outliers
    second_outliers = rm.remove_bedroom_outliers(first_outliers)

    # See the scatter plots after the second removal
    vd.plot_scatter_chart(second_outliers, "Rajaji Nagar")
    vd.plot_scatter_chart(second_outliers, "Hebbal")
    vd.plot_scatter_chart(second_outliers, "Yeshwanthpur")

    data = second_outliers[['total_sqft', 'bath', 'balcony', 'bedroom', 'price_per_sqft']]
    targets = second_outliers['price']

    # Initialize DimensionalityReduction object
    dr = DimensionalityReduction(np.array(data), targets)


    dr.plot_projection(dr.compute_pca(), 'PCA Projection')
