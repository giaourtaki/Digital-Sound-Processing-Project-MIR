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
#print(len(track_id_to_path))
#print(track_id_to_path)

#endregion

#region load audio files

#load mono audio files -> beat tracking, tempo estimation, onset detection, rhythmic analysis, uniform preprocessing
mono_loaded_audio = {}

for track_id, filepath in track_id_to_path.items():
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 1024:
        print(f"Skipping invalid or empty file: {filepath}")
        continue
    try:
        audio = es.MonoLoader(filename=filepath)()
        mono_loaded_audio[track_id] = audio
    except Exception as e:
        print(f"Error loading {filepath} (track_id {track_id}): {e}")
        continue
#test print
#print(len(mono_loaded_audio))
#print(mono_loaded_audio)

#load eqloud audio files -> pitch estimation, music transcription, chord detection, melodic analysis
eqloud_loaded_audio = {}

for track_id, filepath in track_id_to_path.items():
    if not os.path.exists(filepath) or os.path.getsize(filepath) < 1024:
        print(f"Skipping invalid or empty file: {filepath}")
        continue
    try:
        audio = es.EqloudLoader(filename=filepath, sampleRate=44100)()
        eqloud_loaded_audio[track_id] = audio
    except Exception as e:
        print(f"Error loading {filepath} (track_id {track_id}): {e}")
        continue
#test print
#print(len(eqloud_loaded_audio))
#print(eqloud_loaded_audio)
#endregion

#region track feature extracting / processing 

#pitch extraction
def extract_pitch(eqloud_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):

    processed_data_pitch = {}

    window = es.Windowing(type='hann') #Using Hann window 
    yin = es.PitchYin(frameSize=frame_size)

    total_tracks = len(eqloud_loaded_audio)
    for idx, (track_id, audio) in enumerate(eqloud_loaded_audio.items(), start=1):
        pitches = []
        confidences = []
        times = []

        # Process audio in frames with feedback
        duration_sec = len(audio) / sample_rate
        num_frames = (len(audio) - frame_size) // hop_size
        print(f"[{idx}/{total_tracks}] Processing track {track_id} ({duration_sec:.1f}s, ~{num_frames} frames)")

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break

            pitch, confidence = yin(window(frame))
            pitches.append(pitch)
            confidences.append(confidence)
            times.append(i / sample_rate)

            # Log progress every 10 seconds of audio
            if i % (sample_rate * 10) == 0 and i != 0:
                print(f"    ↳ Processed {i / sample_rate:.1f}s")

        processed_data_pitch[track_id] = {
            'pitch_values': pitches,
            'pitch_confidence': confidences,
            'pitch_times': times
        }

    return processed_data_pitch

processed_data_pitch = extract_pitch(eqloud_loaded_audio)
#pitch test print
#print(processed_data_pitch)

#endregion


# Create Features / Target Variables (Make flashcards)



# ML Processing



# Hyperparameter Tuning



# Predictions and Evaluation



# Plot

