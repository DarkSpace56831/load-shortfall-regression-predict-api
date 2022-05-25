"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    
    # Impute Valencia_pressure with the median

    median = feature_vector_df['Valencia_pressure'].median()
    feature_vector_df['Valencia_pressure'].fillna(median, inplace=True)

    # create new features
    feature_vector_df['time']=pd.to_datetime(feature_vector_df['time'], infer_datetime_format=True) 
    feature_vector_df['time_int']= pd.to_numeric(feature_vector_df['time'].dt.strftime("%Y%m%d%H%M%S"))

    #Normalize 23 features
    features = ['Bilbao_pressure', 'Madrid_weather_id', 'Barcelona_weather_id', 'Seville_weather_id',
            'Valencia_pressure', 'Madrid_pressure', 'Bilbao_weather_id', 'Madrid_wind_speed',
            'Bilbao_rain_1h', 'Valencia_wind_speed', 'Bilbao_wind_speed', 'Seville_clouds_all',
            'Barcelona_wind_speed', 'Madrid_clouds_all', 'Seville_wind_speed', 'Barcelona_rain_1h',
            'Seville_rain_1h', 'Bilbao_snow_3h', 'Barcelona_pressure', 'Seville_rain_3h',
            'Madrid_rain_1h', 'Barcelona_rain_3h', 'Valencia_snow_3h']

    from sklearn.preprocessing import MinMaxScaler
    normalize = MinMaxScaler()

    feature_vector_df[features]=normalize.fit_transform(feature_vector_df[features])

    #replacing the string values with the numeric values
    feature_vector_df['Valencia_wind_deg'].replace({'level_5':5, 'level_10':10, 'level_9':9, 'level_8':8, 'level_7':7, 'level_6':6,
       'level_4':4, 'level_3':3, 'level_1':1, 'level_2':2},inplace=True)


    #Replacing the string values in Seville_pressure with numeric values
    feature_vector_df['Seville_pressure'].replace({'sp25':25, 'sp23':23, 'sp24':24, 'sp21':21, 'sp16':16, 'sp9':9, 'sp15':15, 'sp19':19,
       'sp22':22, 'sp11':11, 'sp8':8, 'sp4':4, 'sp6':6, 'sp13':13, 'sp17':17, 'sp20':20,
       'sp18':18, 'sp14':14, 'sp12':12, 'sp5':5, 'sp10':10, 'sp7':7, 'sp3':3, 'sp2':2, 'sp1':1},inplace=True)
    
    # target variable, the variable we want to predict
    y_df_train = feature_vector_df['load_shortfall_3h']

    # features for modeling
    X_df_train = feature_vector_df.drop(labels = ['Unnamed: 0', 'time', 'load_shortfall_3h'], axis = 1)

    #Regularixing the data 
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df_train)
    X_standardise = pd.DataFrame(X_scaled,columns=X_df_train.columns)
    X_standardise.head()

    # create targets and features dataset
    X_train, X_test, y_train, y_test = train_test_split(X_standardise, y_df_train, test_size = 0.3, random_state = 6)

    predict_vector = X_train.copy()
    # ------------------------------------------------------------------------

    return predict_vector

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
