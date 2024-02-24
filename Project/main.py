#%% Visualization
import pandas as pd
import matplotlib.pyplot as plt

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
        :return:
        '''

        #Read the file passed in constructor
        df = pd.read_csv(self.data)

        #Drop unnecessary columns
        df_drop_cols = df.drop(['area_type','society','availability'],axis='columns')

        #Drop null values
        df_drop_null = df_drop_cols.dropna()

        # Return the cleaned DataFrame
        return df_drop_null



if __name__ == '__main__':
    cd = CleanData('bengaluru_house_prices.csv')

    # Capture the cleaned DataFrame
    cleaned_data = cd.clean_data()
