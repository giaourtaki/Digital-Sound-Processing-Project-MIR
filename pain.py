# https://www.youtube.com/watch?v=SW0YGA9d8y8
# https://github.com/microsoft/pylance-release/blob/main/TROUBLESHOOTING.md#unresolved-import-warnings
# source ./soundvenv/bin/activate

# Imports
import os

import IPython.display as ipd
import pandas as pd
import numpy as np
import librosa
import librosa.display
import essentia 


from sklearn.neighbors import KNeighborsClassifier
import sklearn as skl 
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm, sklearn.model_selection, sklearn.metrics 
import matplotlib.pyplot as plt
import seaborn as sns

#import tensorflow as tf
import utils



#region trial

# plt.rcParams['figure.figsize'] = (17, 5)

# # Directory where mp3 are stored.
# AUDIO_DIR = os.environ.get(r"C:\Users\marin\Desktop\Pshfiakh_Texnologia_Hxou\Apallaktikh\Database")

# # Load metadata and features.
# tracks = utils.load('data/fma_metadata/tracks.csv')
# genres = utils.load('data/fma_metadata/genres.csv')
# features = utils.load('data/fma_metadata/features.csv')
# echonest = utils.load('data/fma_metadata/echonest.csv')

# np.testing.assert_array_equal(features.index, tracks.index)
# assert echonest.index.isin(tracks.index).all()

# tracks.shape, genres.shape, features.shape, echonest.shape
#endregion

# Data Frame Creation
data = pd.read_csv('data/fma_metadata/raw_tracks.csv')
# data.info()
# print(data.isnull().sum())




# Data Cleaning and Feature Engineering
def preprocess_data(df):
    # remove irrelevant columns
    df.drop(columns=['album_id', 'album_title', 'album_url', 'artist_id', 'artist_name', 'artist_url', 'artist_website', 'license_image_file', 'license_image_file_large', 'license_parent_id', 'license_title', 'license_url', 'tags', 'track_bit_rate', 'track_comments', 'track_composer', 'track_copyright_c', 'track_copyright_p', 'track_date_created', 'track_date_recorded', 'track_disc_number', 'track_duration', 'track_explicit', 'track_explicit_notes', 'track_favorites', 'track_image_file', 'track_information', 'track_instrumental', 'track_interest', 'track_language_code', 'track_listens', 'track_lyricist', 'track_number', 'track_publisher', 'track_title', 'track_url'], inplace=True)

    # remove empty genre columns
    df.dropna(subset=['track_genres'], inplace=True)
    
    
    return df

preprocess_data(data)
#print(data)
# data.info()
# print(data.isnull().sum())

# Feature Engineering (add essentia stuff)

# first, check which song ids we process
track_ids = data['track_id'].astype(str)
print(track_ids)



# Create Features / Target Variables (Make flashcards)



# ML Processing



# Hyperparameter Tuning



# Predictions and Evaluation



# Plot

