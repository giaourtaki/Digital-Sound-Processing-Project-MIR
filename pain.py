# https://www.youtube.com/watch?v=SW0YGA9d8y8
# https://github.com/microsoft/pylance-release/blob/main/TROUBLESHOOTING.md#unresolved-import-warnings
# source ./soundvenv/bin/activate
# to commit -> move pain.py to rep folder

#region Imports
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

#region track id to path mapping

# first, check which song ids we process
track_ids = data['track_id'].astype(int)

# build a map of song_id -> filepath by walking fma_small
base_dir = 'data/fma_small'
# song_id_to_path to dictionary twn path pou exoun ola ta track paths pou thelw na epeksergastw
track_id_to_path = {}
count = 0
track_limit = 10 #TODO: Limit to 10 tracks for testing; change to 100 for full processing

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

#region yin pitch extraction -> Estimates the fundamental frequency given the frame of a monophonic music signal

def extract_pitch_yin(eqloud_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):

    yin_processed_data_pitch = {}

    window = es.Windowing(type='hann') #Using Hann window 
    yin = es.PitchYin(frameSize=frame_size)

    total_tracks_len = len(eqloud_loaded_audio)
    for idx, (track_id, audio) in enumerate(eqloud_loaded_audio.items(), start=1):
        pitches = []
        confidences = []
        times = []

        # Process audio in frames with feedback
        duration_sec = len(audio) / sample_rate
        num_frames = (len(audio) - frame_size) // hop_size
        print(f"[{idx}/{total_tracks_len}] [Yin] Processing track {track_id} ({duration_sec:.1f}s, ~{num_frames} frames)")

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break

            pitch, confidence = yin(window(frame))
            pitches.append(pitch)
            confidences.append(confidence)
            times.append(i / sample_rate)

            # Log progress every 10 seconds of audio TODO: this doesnt work
            if i % (sample_rate * 10) == 0 and i != 0:
                print(f"    ↳ Processed {i / sample_rate:.1f}s")

        yin_processed_data_pitch[track_id] = {
            'yin_pitch_values': pitches,
            'yin_pitch_confidence': confidences,
            'yin_pitch_times': times
        }

    return yin_processed_data_pitch
#Call the function to extract pitch using Yin
yin_processed_data_pitch = extract_pitch_yin(eqloud_loaded_audio)
#pitch test print
#print(yin_processed_data_pitch)

#endregion

#region yin pitch added to original data
# Add pitch mean as a new feature to the original data DataFrame
data['track_id'] = data['track_id'].astype(int)

# Prepare pitch feature dataframe
yin_pitch_features = []

for track_id, yin_pitch_data in yin_processed_data_pitch.items():
    yin_pitch_values = np.array(yin_pitch_data['yin_pitch_values'])
    yin_confidences = np.array(yin_pitch_data['yin_pitch_confidence'])
    
    # Filter out 0 Hz pitches (no pitch detected)
    valid_yin_pitches = yin_pitch_values[yin_pitch_values > 0]
#TODO: check yin algorithm output types
    
    
    if (len(valid_yin_pitches) > 0):

        yin_mean_confidence_threshold = np.mean(yin_confidences)

       # yin_raw_pitches = valid_yin_pitches
        yin_mean_pitch = np.mean(valid_yin_pitches)
        yin_median_pitch = np.median(valid_yin_pitches)
        yin_std_pitch = np.std(valid_yin_pitches)
    else:
        yin_mean_confidence_threshold = np.nan

        #yin_raw_pitches = np.nan
        yin_mean_pitch = np.nan
        yin_median_pitch = np.nan
        yin_std_pitch = np.nan

    yin_pitch_features.append({
        'track_id': track_id,
       # 'yin_raw_pitches': yin_raw_pitches,
        'yin_pitch_mean': yin_mean_pitch,
        'yin_pitch_median': yin_median_pitch,
        'yin_pitch_std': yin_std_pitch,
        'yin_confidence_threshold': yin_mean_confidence_threshold
    })

# Call the function to create DataFrame and merge pitch_yin with original data
pitch_df = pd.DataFrame(yin_pitch_features)
# Keep only the rows with track_ids in processed_data_pitch
data = data[data['track_id'].isin(yin_processed_data_pitch.keys())]
data = pd.merge(data, pitch_df, on='track_id', how='left')


# Test print
#cols = ['track_id','yin_pitch_median','yin_confidence_threshold']
#print(data[cols])
#print(data)
#print(len(data))
#print(data[['track_id', 'pitch_mean', 'pitch_median', 'pitch_std']].head())
#print("Pitch entries:", list(processed_data_pitch.keys())[:10])
#print("DataFrame track_ids:", data['track_id'].head(10).tolist())
#print(data.info())
#endregion

#region melody / predominant pitch extraction -> Estimates the fundamental frequency of the predominant melody from polyphonic music signals using the MELODIA algorithm
def extract_pitch_melodia(eqloud_loaded_audio, frame_size=2048, hop_size=128, sample_rate=44100):

    processed_data_pitch_melodia = {}


    melodia = es.PredominantPitchMelodia(frameSize=frame_size, hopSize=hop_size)
    

    
    for idx, (track_id, audio) in enumerate(eqloud_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        num_frames = (len(audio) - frame_size) // hop_size
        total_tracks = len(eqloud_loaded_audio)
        print(f"[{idx}/{total_tracks}] [Melodia] Processing track {track_id} ({duration_sec:.1f}s, ~{num_frames} frames)")

        try:
            pitch_values, pitch_confidence = melodia(audio)
            pitch_times = np.arange(len(pitch_values)) * hop_size / sample_rate

            processed_data_pitch_melodia[track_id] = {
                'pitch_values': pitch_values,
                'pitch_confidence': pitch_confidence,
                'pitch_times': pitch_times
            }

        except Exception as e:
            print(f"Error processing track {track_id}: {e}")
            continue

    return processed_data_pitch_melodia

#Call the function to extract pitch using Melodia
#processed_data_pitch_melodia = extract_pitch_melodia(eqloud_loaded_audio)

#Test print
#print(processed_data_pitch_melodia)
#endregion

#region melodia pitch added to original data


#endregion


# Create Features / Target Variables (Make flashcards)



# ML Processing



# Hyperparameter Tuning



# Predictions and Evaluation



# Plot

