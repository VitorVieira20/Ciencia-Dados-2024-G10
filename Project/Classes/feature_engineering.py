class FeatureEngineering:
    """
    A class for feature engineering on the dataset
    """

    def __init__(self, data_loader):
        """
        Initializes the FeatureEngineering object

        :param data_loader: The input dataset
        """
        self.data_loader = data_loader
        self.data_train = None
        self.labels_train = None
        self.data_test = None
        self.labels_test = None

    def create_features(self):
        """
        Creates new features based on the existing data
        """
        self.data_train = self.data_loader.data_train
        self.labels_train = self.data_loader.labels_train
        self.data_test = self.data_loader.data_test
        self.labels_test = self.data_loader.labels_test

        # Price distribution per bedroom.
        self.data_train['price_per_bedroom'] = self.data_train['price'] / self.data_train['rooms']
        self.data_test['price_per_bedroom'] = self.data_test['price'] / self.data_test['rooms']

        # Bathroom to bedroom ratio.
        self.data_train['bath_per_bedroom'] = self.data_train['bath'] / self.data_train['rooms']
        self.data_test['bath_per_bedroom'] = self.data_test['bath'] / self.data_test['rooms']

        # Space distribution per bedroom.
        self.data_train['total_sqft_per_room'] = self.data_train['total_sqft'] / self.data_train['rooms']
        self.data_test['total_sqft_per_room'] = self.data_test['total_sqft'] / self.data_test['rooms']

        # Balcony to bedroom ratio
        self.data_train['balcony_per_room'] = self.data_train['balcony'] / self.data_train['rooms']
        self.data_test['balcony_per_room'] = self.data_test['balcony'] / self.data_test['rooms']

        # Price distribution per bathroom
        self.data_train['price_per_bath'] = self.data_train['price'] / self.data_train['bath']
        self.data_test['price_per_bath'] = self.data_test['price'] / self.data_test['bath']

        # Space distribution per bathroom
        self.data_train['total_sqft_per_bath'] = self.data_train['total_sqft'] / self.data_train['bath']
        self.data_test['total_sqft_per_bath'] = self.data_test['total_sqft'] / self.data_test['bath']

        # Balcony to bathroom ratio
        self.data_train['balcony_per_bath'] = self.data_train['balcony'] / self.data_train['bath']
        self.data_test['balcony_per_bath'] = self.data_test['balcony'] / self.data_test['bath']

        # Price distribution per balcony
        self.data_train['price_per_balcony'] = self.data_train['price'] / self.data_train['balcony']
        self.data_test['price_per_balcony'] = self.data_test['price'] / self.data_test['balcony']

        # Space distribution per balcony
        self.data_train['total_sqft_per_balcony'] = self.data_train['total_sqft'] / self.data_train['balcony']
        self.data_test['total_sqft_per_balcony'] = self.data_test['total_sqft'] / self.data_test['balcony']

        # Bedroom to bathroom ratio
        self.data_train['room_per_bath'] = self.data_train['rooms'] / self.data_train['bath']
        self.data_test['room_per_bath'] = self.data_test['rooms'] / self.data_test['bath']

        # Calculate the average price
        average_price_train = self.data_train['price'].mean()
        average_price_test = self.data_test['price'].mean()

        # Prices above average
        self.data_train['price_above_average'] = self.data_train['price'] > average_price_train
        self.data_test['price_above_average'] = self.data_test['price'] > average_price_test