import pandas as pd
import numpy as np
import time

# Classes
from Project.Classes.data_loader import DataLoader
from Project.Classes.clean_data import CleanData
from Project.Classes.filter_locations import FilterLocations
from Project.Classes.Outliers.price_sqft_outliers import RemovePriceBySquareFeatOutliers
from Project.Classes.Outliers.price_location_outliers import RemovePriceLocationsOutliers
from Project.Classes.Outliers.bedrooms_outliers import RemoveBedroomsOutliers
from Project.Classes.Outliers.remove_remaining_outliers import RemoveRemainingOutliers
from Project.Classes.dimensionality_reduction import DimensionalityReduction
from Project.Classes.hypothesis_tester import HypothesisTester
from Project.Classes.feature_engineering import FeatureEngineering
from Project.Classes.model_selection import ModelSelection

# Functions
from Project.Classes.shared_functions import print_shape
from Project.Classes.shared_functions import plot_data_visualizations
from Project.Classes.shared_functions import print_hypothesis_result

#####################################################################
#                       LOAD AND CLEAN DATA                         #
#####################################################################
# Load data
data_loader = DataLoader("bengaluru_house_prices.csv")
print_shape(data_loader, "Raw Data")

# First step for cleaning data
clean_data = CleanData(data_loader)
clean_data.clean_data()
print_shape(clean_data, "Clean Data")

# Clean data based on locations
filter_locations = FilterLocations(clean_data)
filter_locations.clean_data()
print_shape(filter_locations, "Clean Data by Locations")

# Remove outliers based on price per square feat
remove_price_by_square_feat = RemovePriceBySquareFeatOutliers(filter_locations)
remove_price_by_square_feat.clean_data()
print_shape(remove_price_by_square_feat, "Remove outliers based on Price by Square Feat")

# Remove outliers based on price and location
remove_price_location = RemovePriceLocationsOutliers(remove_price_by_square_feat)
remove_price_location.clean_data()
print_shape(remove_price_location, "Remove outliers based on Price and Locations")

# Remove outliers based on bedrooms and total square feat
remove_bedroom = RemoveBedroomsOutliers(remove_price_location)
remove_bedroom.clean_data()
print_shape(remove_bedroom, "Remove outliers based on Bedrooms and Total Square Feat")

# Remove remaining outliers
remove_outliers = RemoveRemainingOutliers(remove_bedroom)
remove_outliers.remove_outliers()
print_shape(remove_outliers, "Remove remaining outliers")



#####################################################################
#                       VISUALIZE PLOTS                             #
#####################################################################

# Show data from Remove outliers based on price per square feat
plot_data_visualizations(remove_price_by_square_feat.data_train, ['Electronic City', 'Raja Rajeshwari Nagar', 'Sarjapur  Road'])
time.sleep(40)

# Show data from Remove outliers based on price and location
plot_data_visualizations(remove_price_location.data_train, ['Electronic City', 'Raja Rajeshwari Nagar', 'Sarjapur  Road'])
time.sleep(40)

# Show data from Remove outliers based on bedrooms and total square feat
plot_data_visualizations(remove_bedroom.data_train, ['Electronic City', 'Raja Rajeshwari Nagar', 'Sarjapur  Road'])
time.sleep(40)

# Show data from Remove Remaining outliers
plot_data_visualizations(remove_outliers.data_train, ['Electronic City', 'Raja Rajeshwari Nagar', 'Sarjapur  Road'])
time.sleep(40)



#####################################################################
#                    DIMENSIONAL REDUCTIONS                         #
#####################################################################

# Our data
data = remove_outliers.data_train.drop(columns=["price", "location"])

# Our target
targets = remove_outliers.labels_train

# Initialize DimensionalityReduction object
dr = DimensionalityReduction(data, targets)

# Show PCA Projection
dr.plot_projection(dr.compute_pca(), 'PCA Projection')

# Show UMAP Projection
dr.plot_projection(dr.compute_umap(), 'UMAP Projection')



#####################################################################
#                        HYPOTHESIS TEST                            #
#####################################################################

# Prices per SQq. feet for three locations
data = remove_outliers.data_train

price_sqft_Electronic_City = data[data['location'] == 'Electronic City']['price_per_sqft']
price_sqft_Raja_Nagar = data[data['location'] == 'Raja Rajeshwari Nagar']['price_per_sqft']
price_sqft_Sarjapur_Road = data[data['location'] == 'Sarjapur  Road']['price_per_sqft']

# Prices for Rajaji Nagar location
price_Electronic_City = data[data['location'] == 'Electronic City']['price']

# Total Sq. Feet for Rajaji Nagar location
total_sqft_Raja_Nagar = data[data['location'] == 'Raja Rajeshwari Nagar']['total_sqft']

tester = HypothesisTester()

# Perform unpaired t-test between Electronic City and Raja Rajeshwari Nagar locations
t_stat, p_val = tester.unpaired_t_test(price_Electronic_City, price_sqft_Raja_Nagar)
print_hypothesis_result("Unpaired t-test between Electronic City and Raja Rajeshwari Nagar locations:", "t-statistic", t_stat, "p-value", p_val)

# Perform unpaired ANOVA among the three locations
f_stat, p_val_anova = tester.unpaired_anova(price_sqft_Raja_Nagar, price_sqft_Electronic_City, price_sqft_Sarjapur_Road)
print_hypothesis_result("Unpaired ANOVA among three locations:", "F-statistic", f_stat, "p-value", p_val_anova)

# Perform paired t-test for total_sqft and price within Raja Rajeshwari Nagar location
t_stat_paired, p_val_paired = tester.paired_t_test(total_sqft_Raja_Nagar, price_sqft_Raja_Nagar)
print_hypothesis_result("Paired t-test for total_sqft and price within Raja Rajeshwari location:", "t-statistic", t_stat_paired, "p-value", p_val_paired)



#####################################################################
#                       FEATURE ENGINEERING                         #
#####################################################################

# Create new features based on feature engineering
new_features = FeatureEngineering(remove_outliers)
new_features.create_features()
print_shape(new_features, "Data with new features")



#####################################################################
#                         MODEL SELECTION                           #
#####################################################################

new_features_encoded = pd.get_dummies(new_features.data_train)
new_features_encoded.fillna(new_features_encoded.mean(), inplace=True)

for column in new_features_encoded.columns:
    if np.isinf(new_features_encoded[column]).any():
        max_val = new_features_encoded[~np.isinf(new_features_encoded[column])][column].max()
        new_features_encoded[column] = new_features_encoded[column].replace([np.inf, -np.inf], max_val)

# Now, 'new_features_encoded' is your DataFrame with encoded categorical variables
X = new_features_encoded.drop(['price'], axis=1)
y = new_features_encoded['price']

# Initialize the ModelSelection class with the data
ms = ModelSelection(X, y)

# Perform model selection
ms.select_model()