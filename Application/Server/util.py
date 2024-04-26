import json
import pickle
import numpy as np
import pandas as pd
import os

# If the best model is KNN, it will be necessary this import
from Project.Classes.ModelClasses.KNNRegression import KNNRegression

__locations = None
__data_columns = None
__model = None

def get_predicted_price(location, total_sqft, rooms, bath, balcony):
    predict_df = pd.DataFrame(np.zeros((1, len(__data_columns)), dtype=int), columns=__data_columns)

    predict_df.at[0, 'total_sqft'] = total_sqft
    predict_df.at[0, 'bath'] = bath
    predict_df.at[0, 'balcony'] = balcony
    predict_df.at[0, 'rooms'] = rooms
    predict_df.at[0, 'area_per_room'] = total_sqft / rooms
    predict_df.at[0, 'area_per_bath'] = total_sqft / bath
    predict_df.at[0, 'area_per_balcony'] = total_sqft / balcony
    predict_df.at[0, 'room_bath_ratio'] = rooms / bath
    predict_df.at[0, 'room_balcony_ratio'] = rooms / balcony
    predict_df.at[0, 'bath_balcony_ratio'] = bath / balcony
    predict_df.at[0, 'average_length_per_room'] = (total_sqft * 0.092903) / rooms
    predict_df.at[0, 'average_length_per_bath'] = (total_sqft * 0.092903) / bath
    predict_df.at[0, 'average_length_per_balcony'] = (total_sqft * 0.092903) / balcony
    predict_df.at[0, 'total_room_bath'] = rooms + bath
    predict_df.at[0, location] = 1

    return round(__model.predict(predict_df)[0], 2)

def get_location_names():
    return __locations

def load_columns():
    print('Loading data...')
    global __locations
    global __data_columns
    global __model

    # Folder created in the Jupyter Notebook or another with the same structure
    models_dir = '../../Project/Models1/'

    data_columns_path = models_dir + 'data_columns.json'

    # Search for Best Model folder
    best_model_dir = None
    for dir_name in os.listdir(models_dir):
        if dir_name.startswith('Best') and os.path.isdir(os.path.join(models_dir, dir_name)):
            best_model_dir = os.path.join(models_dir, dir_name)
            break

    if best_model_dir is not None:
        print("Diretório do melhor modelo encontrado:", best_model_dir)
    else:
        print("Nenhum diretório começando com 'Best' encontrado em", models_dir)

    best_model_path = best_model_dir + '/best_model.pkl'
    print(best_model_path)

    with open(data_columns_path, 'r') as f:
        __data_columns = json.loads(f.read())
        __locations = __data_columns[14:]

    with open(best_model_path, 'rb') as f:
        __model = pickle.load(f)
        
    print('Done!')

if __name__ == '__main__':
    load_columns()
    print(get_location_names())