import pandas as pd

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

    def _clean_data(self):
        '''
        Cleans the data
        :return:
        '''

        #Read the file passed in constructor
        housing_data = pd.read_csv(self.data)

        #Drop null values
        housing_data.dropna()

        #Drop duplicated values
        housing_data.drop_duplicates()

        print(housing_data)



if __name__ == '__main__':
    cd = CleanData('ParisHousing.csv')

    cd._clean_data()