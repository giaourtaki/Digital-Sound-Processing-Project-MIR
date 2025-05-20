# https://www.youtube.com/watch?v=SW0YGA9d8y8
# https://github.com/microsoft/pylance-release/blob/main/TROUBLESHOOTING.md#unresolved-import-warnings
# source ./soundvenv/bin/activate
# to commit -> move pain.py to rep folder

# Imports
import os

import IPython.display as ipd
import pandas as pd
import numpy as np

import librosa
import librosa.display
import essentia 
import essentia.standard as es


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

#region edw epilegw poia tragoudia epeksergazomai
# first, check which song ids we process
track_ids = data['track_id'].astype(str)

# build a map of song_id -> filepath by walking fma_small
base_dir = 'data/fma_small'
# song_id_to_path to dictionary twn path pou exoun ola ta track paths pou thelw na epeksergastw
track_id_to_path = {}
count = 0
track_limit = 100

# Walk through all subfolders and collect matching file paths
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith('.mp3'):
            track_id = int(file.replace('.mp3', ''))
            if track_id in track_ids:
                full_path = os.path.join(root, file)
                track_id_to_path[track_id] = full_path
                count += 1
                if count >= track_limit:
                    break
    if count >= track_limit:
        break
#test print
print(len(track_id_to_path))
#print(track_id_to_path)

#endregion

#region load audio files
loaded_audio = {}
#TODO : check if the audio files are loaded correctly

for track_id, filepath in track_id_to_path.items():
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 1024:
        print(f"⚠️ Skipping invalid or empty file: {filepath}")
        continue
    try:
        audio = es.MonoLoader(filename=filepath)()
        loaded_audio[track_id] = audio
    except Exception as e:
        print(f"❌ Error loading {filepath} (track_id {track_id}): {e}")
        continue
#test print
print(len(loaded_audio))
print(loaded_audio)
#endregion

#region track feature extracting / processing 
processed_data_pitch = []

for track_id, filepath in track_id_to_path.items():

    audio = es.MonoLoader(filename=filepath)()
    
    #features = extract_features(filepath)  # Your feature extractor
    #features['song_id'] = int(track_id)
    #processed_data_pitch.append(features)
#endregion


# Create Features / Target Variables (Make flashcards)



# ML Processing



# Hyperparameter Tuning



# Predictions and Evaluation



# Plot

