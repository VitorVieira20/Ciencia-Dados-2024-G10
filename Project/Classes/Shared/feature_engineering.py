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

        # Area per room: Divides the total area by the number of rooms.
        self.data_train['area_per_room'] = self.data_train['total_sqft'] / self.data_train['rooms']
        self.data_test['area_per_room'] = self.data_test['total_sqft'] / self.data_test['rooms']

        # Area per bathroom: Divides the total area by the number of bathrooms.
        self.data_train['area_per_bath'] = self.data_train['total_sqft'] / self.data_train['bath']
        self.data_test['area_per_bath'] = self.data_test['total_sqft'] / self.data_test['bath']

        # Area per balcony: Divides the total area by the number of balconies.
        self.data_train['area_per_balcony'] = self.data_train['total_sqft'] / self.data_train['balcony']
        self.data_test['area_per_balcony'] = self.data_test['total_sqft'] / self.data_test['balcony']

        # Room-to-bathroom ratio: Divides the number of rooms by the number of bathrooms.
        self.data_train['room_bath_ratio'] = self.data_train['rooms'] / self.data_train['bath']
        self.data_test['room_bath_ratio'] = self.data_test['rooms'] / self.data_test['bath']

        # Room-to-balcony ratio: Divides the number of rooms by the number of balconies.
        self.data_train['room_balcony_ratio'] = self.data_train['rooms'] / self.data_train['balcony']
        self.data_test['room_balcony_ratio'] = self.data_test['rooms'] / self.data_test['balcony']

        # Bathroom-to-balcony ratio: Divides the number of bathrooms by the number of balconies.
        self.data_train['bath_balcony_ratio'] = self.data_train['bath'] / self.data_train['balcony']
        self.data_test['bath_balcony_ratio'] = self.data_test['bath'] / self.data_test['balcony']

        # Average length per room in square meters.
        self.data_train['average_length_per_room'] = (self.data_train['total_sqft'] * 0.092903) / self.data_train[
            'rooms']
        self.data_test['average_length_per_room'] = (self.data_test['total_sqft'] * 0.092903) / self.data_test['rooms']

        # Average length per bathroom in square meters.
        self.data_train['average_length_per_bath'] = (self.data_train['total_sqft'] * 0.092903) / self.data_train[
            'bath']
        self.data_test['average_length_per_bath'] = (self.data_test['total_sqft'] * 0.092903) / self.data_test['bath']

        # Average length per balcony in square meters.
        self.data_train['average_length_per_balcony'] = (self.data_train['total_sqft'] * 0.092903) / self.data_train[
            'balcony']
        self.data_test['average_length_per_balcony'] = (self.data_test['total_sqft'] * 0.092903) / self.data_test[
            'balcony']

        # Total number of rooms and bathrooms.
        self.data_train['total_room_bath'] = self.data_train['rooms'] + self.data_train['bath']
        self.data_test['total_room_bath'] = self.data_test['rooms'] + self.data_test['bath']