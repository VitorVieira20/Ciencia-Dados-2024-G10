import pandas as pd
import numpy as np

# Classes
from Project.Classes.data_loader import DataLoader
from Project.Classes.clean_data import CleanData
from Project.Classes.filter_locations import FilterLocations
from Project.Classes.price_sqft_outliers import RemovePriceBySquareFeatOutliers
from Project.Classes.price_location_outliers import RemovePriceLocationsOutliers
from Project.Classes.bedrooms_outliers import RemoveBedroomsOutliers
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
removePriceBySquareFeat = RemovePriceBySquareFeatOutliers(filter_locations)
removePriceBySquareFeat.clean_data()
print_shape(removePriceBySquareFeat, "Remove outliers based on Price by Square Feat")

# Remove outliers based on price and location
removePriceLocation = RemovePriceLocationsOutliers(removePriceBySquareFeat)
removePriceLocation.clean_data()
print_shape(removePriceLocation, "Remove outliers based on Price and Locations")

# Remove outliers based on bedrooms and total square feat
removeBedroom = RemoveBedroomsOutliers(removePriceLocation)
removeBedroom.clean_data()
print_shape(removeBedroom, "Remove outliers based on Bedrooms and Total Square Feat")



#####################################################################
#                       VISUALIZE PLOTS                             #
#####################################################################

# Show data from Remove outliers based on price per square feat
plot_data_visualizations(removePriceBySquareFeat.data_train, ["Rajaji Nagar", "Hebbal", "Yeshwanthpur"])

# Show data from Remove outliers based on price and location
plot_data_visualizations(removePriceLocation.data_train, ["Rajaji Nagar", "Hebbal", "Yeshwanthpur"])

# Show data from Remove outliers based on bedrooms and total square feat
plot_data_visualizations(removeBedroom.data_train, ["Rajaji Nagar", "Hebbal", "Yeshwanthpur"])



#####################################################################
#                    DIMENSIONAL REDUCTIONS                         #
#####################################################################

# Our data
data = removeBedroom.data_train.drop(columns=["price", "location"])

# Our target
targets = removeBedroom.labels_train

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
data = removeBedroom.data_train

price_sqft_Rajaji_Nagar = data[data['location'] == 'Rajaji Nagar']['price_per_sqft']
price_sqft_Hebbal = data[data['location'] == 'Hebbal']['price_per_sqft']
price_sqft_Yeshwanthpur = data[data['location'] == 'Yeshwanthpur']['price_per_sqft']

# Prices for Rajaji Nagar location
price_Rajaji_Nagar = data[data['location'] == 'Rajaji Nagar']['price']

# Total Sq. Feet for Rajaji Nagar location
total_sqft_Rajaji_Nagar = data[data['location'] == 'Rajaji Nagar']['total_sqft']

tester = HypothesisTester()

# Perform unpaired t-test between Hebbal and Rajaji Nagar locations
t_stat, p_val = tester.unpaired_t_test(price_sqft_Hebbal, price_sqft_Rajaji_Nagar)
print_hypothesis_result("Unpaired t-test between Hebbal and Rajaji Nagar locations:", "t-statistic", t_stat, "p-value", p_val)

# Perform unpaired ANOVA among the three locations
f_stat, p_val_anova = tester.unpaired_anova(price_sqft_Rajaji_Nagar, price_sqft_Hebbal, price_sqft_Yeshwanthpur)
print_hypothesis_result("Unpaired ANOVA among three locations:", "F-statistic", f_stat, "p-value", p_val_anova)

# Perform paired t-test for total_sqft and price within Rajaji Nagar location
t_stat_paired, p_val_paired = tester.paired_t_test(total_sqft_Rajaji_Nagar, price_Rajaji_Nagar)
print_hypothesis_result("Paired t-test for total_sqft and price within Rajaji Nagar location:", "t-statistic", t_stat_paired, "p-value", p_val_paired)



#####################################################################
#                       FEATURE ENGINEERING                         #
#####################################################################

# Create new features based on feature engineering
newFeatures = FeatureEngineering(removeBedroom)
newFeatures.create_features()
print_shape(newFeatures, "Data with new features")



#####################################################################
#                         MODEL SELECTION                           #
#####################################################################

new_features_encoded = pd.get_dummies(newFeatures.data_train)
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