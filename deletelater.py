# https://www.youtube.com/watch?v=SW0YGA9d8y8
# https://github.com/microsoft/pylance-release/blob/main/TROUBLESHOOTING.md#unresolved-import-warnings
# source ./soundvenv/bin/activate
# to commit -> move pain.py to rep folder
#static analysis -> mean, median, std, skewness, kurtosis, slope, root mean square, first and second derivative. 

#region Imports
import os
import csv
import json

import IPython.display as ipd
import pandas as pd
import numpy as np
from numpy import pad

import librosa
#import librosa.display
#import essentia 
import essentia.standard as es
from scipy.signal import find_peaks
import scipy 
from scipy import stats
from scipy.stats import skew, kurtosis



from sklearn.neighbors import KNeighborsClassifier
import sklearn as skl 
import sklearn.utils, sklearn.preprocessing, sklearn.decomposition, sklearn.svm, sklearn.model_selection, sklearn.metrics 
import matplotlib.pyplot as plt
import seaborn as sns

#import tensorflow as tf
import utils
import sqlite3
connection = sqlite3.connect('sound_database.db')
#endregion

# Data Frame Creation
data = pd.read_csv('data/fma_metadata/raw_tracks.csv')
# data.info()
# print(data.isnull().sum())



# Data Cleaning and Feature Engineering
def preprocess_data(df):
    # remove irrelevant columns
    df.drop(columns=['album_id', 'album_title', 'album_url', 'track_file', 'artist_id', 'artist_name', 'artist_url', 'artist_website', 'license_image_file', 'license_image_file_large', 'license_parent_id', 'license_title', 'license_url', 'tags', 'track_bit_rate', 'track_comments', 'track_composer', 'track_copyright_c', 'track_copyright_p', 'track_date_created', 'track_date_recorded', 'track_disc_number', 'track_duration', 'track_explicit', 'track_explicit_notes', 'track_favorites', 'track_image_file', 'track_information', 'track_instrumental', 'track_interest', 'track_language_code', 'track_listens', 'track_lyricist', 'track_number', 'track_publisher', 'track_title', 'track_url'], inplace=True)

    # remove empty genre columns
    df.dropna(subset=['track_genres'], inplace=True)
    
    
    return df

data = preprocess_data(data)
#print(data)
# data.info()
# print(data.isnull().sum())

# Feature Engineering (add essentia stuff)

#region track id to path mapping


def get_fma_small_ids(base_dir):
    valid_ids = set()
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.mp3'):
                try:
                    tid = int(file.replace('.mp3', ''))
                    valid_ids.add(tid)
                except ValueError:
                    continue
    return valid_ids
base_dir = 'data/fma_small'
valid_track_ids = get_fma_small_ids(base_dir)


# first, check which song ids we process
track_ids = data['track_id'].astype(int)
genres = data['track_genres']
#print(genres)

# Define genre ID mapping
genre_ids = {
    '10': 'Pop',
    '12': 'Rock',
    '31': 'Metal',
    '25': 'Punk'
}

# Set limits for each genre TODO: change this to 80 songs per genre
genre_limits = {
    'Pop': 1,
    'Rock': 1,
    'Metal': 1,
    'Punk': 1
}
genre_counts = {
    'Pop': 0,
    'Rock': 0,
    'Metal': 0,
    'Punk': 0
}

track_id_to_genre = {}

# Iterate through DataFrame to find tracks by genre
for idx, row in data.iterrows():
    tid = int(row['track_id'])
    if tid not in valid_track_ids:
        continue  # Skip if track ID is not in valid set
    genre_str = row['track_genres']
    for genre_id, genre_name in genre_ids.items():
        if f'"genre_id": "{genre_id}"' in genre_str or f"'genre_id': '{genre_id}'" in genre_str:
            if genre_counts[genre_name] < genre_limits[genre_name]:
                track_id_to_genre[int(row['track_id'])] = genre_name
                genre_counts[genre_name] += 1
            break  # Only count the track once

#Test print
print(f"Collected {len(track_id_to_genre)} tracks:")
print("track_id_to_genre:", track_id_to_genre)
#for tid, path in track_id_to_genre.items():
#print(f"Track ID: {tid}, Genre: {track_id_to_genre[tid]}")


# Build path map from matched track_ids 
track_id_to_path_another_name = {}

def collect_track_paths(base_dir, track_id_to_genre):
    track_id_to_path = {}
    
    # Convert track IDs to a set for fast lookup
    target_ids = set(track_id_to_genre.keys())
    
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith('.mp3'):
                try:
                    # Extract track ID from filename (e.g., '000123.mp3' -> 123)
                    track_id = int(file.replace('.mp3', ''))
                    
                    if track_id in target_ids:
                        full_path = os.path.join(root, file)
                        track_id_to_path[track_id] = full_path
                except ValueError:
                    # In case the filename is not a valid int
                    continue
    
   # print(f"Found paths for {len(track_id_to_path)} out of {len(track_id_to_genre)} tracks.")
   
    return track_id_to_path

track_id_to_path_another_name = collect_track_paths(base_dir, track_id_to_genre)
# Test print
print(f"Collected {len(track_id_to_path_another_name)} tracks:")
#missing_ids = set(track_id_to_genre.keys()) - set(track_id_to_path.keys())
#print("Missing track IDs (no .mp3 file found):", missing_ids)
#for tid, path in track_id_to_path.items():
  #  print(f"Track ID: {tid}, Genre: {track_id_to_genre[tid]}, Path: {path}")
#print(len(track_id_to_path))
#print(track_id_to_path)

#endregion



"""
=====================================
|        Melody Features          |
=====================================
"""
#region Yin Pitch Extraction -> Estimates the fundamental frequency given the frame of a monophonic music signal

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
        print(f"[{idx}/{total_tracks_len}] [Yin Pitch] Processing track {track_id} ({duration_sec:.1f}s, ~{num_frames} frames)")

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break

            pitch, confidence = yin(window(frame))
            pitches.append(pitch)
            confidences.append(confidence)
            times.append(i / sample_rate)

    

        yin_processed_data_pitch[track_id] = {
            'yin_pitch_values': pitches,
            'yin_pitch_confidence': confidences,
            'yin_pitch_times': times
        }

    return yin_processed_data_pitch

#pitch test print
#print(yin_processed_data_pitch)



#endregion

#region Melody / Predominant pitch Extraction -> Estimates the fundamental frequency of the predominant melody from polyphonic music signals using the MELODIA algorithm
def extract_pitch_melodia(eqloud_loaded_audio, frame_size=2048, hop_size=128, sample_rate=44100): 

    processed_data_pitch_melodia = {}


    melodia = es.PredominantPitchMelodia(frameSize=frame_size, hopSize=hop_size)
    

    
    for idx, (track_id, audio) in enumerate(eqloud_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        num_frames = (len(audio) - frame_size) // hop_size
        total_tracks = len(eqloud_loaded_audio)
        print(f"[{idx}/{total_tracks}] [Melodia Pitch] Processing track {track_id} ({duration_sec:.1f}s, ~{num_frames} frames)")

        try:
            melodia_pitch_values, melodia_pitch_confidence = melodia(audio)
            melodia_pitch_times = np.arange(len(melodia_pitch_values)) * hop_size / sample_rate

            processed_data_pitch_melodia[track_id] = {
                'melodia_pitch_values': melodia_pitch_values,
                'melodia_pitch_confidence': melodia_pitch_confidence,
                'melodia_pitch_times': melodia_pitch_times,
                'melodia_pitch_mean': np.mean(melodia_pitch_values),
                'melodia_pitch_median': np.median(melodia_pitch_values),
                'melodia_pitch_std': np.std(melodia_pitch_values),
                'melodia_pitch_skewness': skew(melodia_pitch_values),
                'melodia_pitch_kurtosis': kurtosis(melodia_pitch_values),
                'melodia_pitch_rms': np.sqrt(np.mean(np.square(melodia_pitch_values))),
                'melodia_pitch_delta': np.mean(np.diff(melodia_pitch_values)) if len(melodia_pitch_values) > 1 else 0
            }
            # Test print 
            #print(f"[Melodia Pitch Stats] Track {track_id}: {processed_data_pitch_melodia[track_id]}")

        except Exception as e:
            print(f"Error processing track {track_id}: {e}")
            continue

        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, melodia_pitch_mean, melodia_pitch_median, melodia_pitch_std, melodia_pitch_skewness, melodia_pitch_kurtosis, melodia_pitch_rms, melodia_pitch_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    melodia_pitch_mean = excluded.melodia_pitch_mean,
                    melodia_pitch_median = excluded.melodia_pitch_median,
                    melodia_pitch_std = excluded.melodia_pitch_std,
                    melodia_pitch_skewness = excluded.melodia_pitch_skewness, 
                    melodia_pitch_kurtosis = excluded.melodia_pitch_kurtosis , 
                    melodia_pitch_rms = excluded.melodia_pitch_rms, 
                    melodia_pitch_delta = excluded.melodia_pitch_delta"""
        # Get the correct values from the processed_data_pitch_melodia dict for this track
        vals = processed_data_pitch_melodia[track_id]
        cursor.execute(sql, (
            track_id,
            float(vals['melodia_pitch_mean']),
            float(vals['melodia_pitch_median']),
            float(vals['melodia_pitch_std']),
            float(vals['melodia_pitch_skewness']),
            float(vals['melodia_pitch_kurtosis']),
            float(vals['melodia_pitch_rms']),
            float(vals['melodia_pitch_delta'])
        ))
        connection.commit()

    return processed_data_pitch_melodia




#endregion


#region Melodia Pitch Range Extraction
def extract_melodic_pitch_range(eqloud_loaded_audio, frame_size=2048, hop_size=128, sample_rate=44100, min_confidence=0.1):

    melodic_pitch_range_data = {}
    melodia = es.PredominantPitchMelodia(frameSize=frame_size, hopSize=hop_size)

    for idx, (track_id, audio) in enumerate(eqloud_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        total_tracks = len(eqloud_loaded_audio)
        print(f"[{idx}/{total_tracks}] [Melodia Pitch Range] Processing track {track_id} ({duration_sec:.1f}s)")

        try:
            pitch_values, pitch_confidence = melodia(audio)

            # Filter out invalid (0) pitch values and those below confidence threshold
            valid_pitches = [p for p, c in zip(pitch_values, pitch_confidence) if p > 0 and c >= min_confidence]

            if valid_pitches:
                pitch_range = float(np.max(valid_pitches) - np.min(valid_pitches))
            else:
                pitch_range = 0.0  # No valid pitch values detected

            melodic_pitch_range_data[track_id] = {
                'melodic_pitch_range': pitch_range
            }

        except Exception as e:
            print(f"Error processing track {track_id}: {e}")
            continue
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, melodic_pitch_range) values (?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    melodic_pitch_range = excluded.melodic_pitch_range"""
        vals = melodic_pitch_range_data[track_id]     
        cursor.execute(sql, (
        track_id,
        float(vals['melodic_pitch_range'])
        ))
        connection.commit()

    return melodic_pitch_range_data

# Test print
#print(processed_data_melodic_pitch_range)
#endregion

#region MIDI Note Number (MNN) Statistics 
def extract_mnn_stats(processed_data_pitch_melodia):
    
    processed_data_mnn = {}

    for track_id, pitch_data in processed_data_pitch_melodia.items():
        freqs = np.array(pitch_data['melodia_pitch_values'])
        # Keep only frames with a detected pitch > 0 Hz
        freqs = freqs[freqs > 0]

        if freqs.size > 0:
            # Convert Hz -> MIDI note numbers (float)
            mnn = 69 + 12 * np.log2(freqs / 440.0)
            # Round to nearest integer semitone
            mnn_int = np.round(mnn).astype(int)

            processed_data_mnn[track_id] = {
                'mnn_mean':   np.mean(mnn_int),
                'mnn_median': np.median(mnn_int),
                'mnn_std':    np.std(mnn_int),
                'mnn_skewness': skew(mnn_int),
                'mnn_kurtosis': kurtosis(mnn_int),
                'mnn_rms': np.sqrt(np.mean(np.square(mnn_int))),
                'mnn_delta': np.mean(np.diff(mnn_int)) if len(mnn_int) > 1 else 0
                
            }
            # Test print 
            #print(f"[MNN Stats] Track {track_id}: {processed_data_mnn[track_id]}") 
        else:
            # No valid pitches detected
            processed_data_mnn[track_id] = {
                'mnn_mean':   np.nan,
                'mnn_median': np.nan,
                'mnn_std':    np.nan,
                'mnn_skewness': np.nan,
                'mnn_kurtosis': np.nan,
                'mnn_rms': np.nan,
                'mnn_delta': np.nan
            }

        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, mnn_mean, mnn_median, mnn_std, mnn_skewness, mnn_kurtosis, mnn_rms, mnn_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    mnn_mean = excluded.mnn_mean,
                    mnn_median = excluded.mnn_median,
                    mnn_std = excluded.mnn_std,
                    mnn_skewness = excluded.mnn_skewness, 
                    mnn_kurtosis = excluded.mnn_kurtosis , 
                    mnn_rms = excluded.mnn_rms, 
                    mnn_delta = excluded.mnn_delta"""
        # Get the correct values from the processed_data_pitch_melodia dict for this track
        vals = processed_data_mnn[track_id]
        cursor.execute(sql, (
            track_id,
            float(vals['mnn_mean']),
            float(vals['mnn_median']),
            float(vals['mnn_std']),
            float(vals['mnn_skewness']),
            float(vals['mnn_kurtosis']),
            float(vals['mnn_rms']),
            float(vals['mnn_delta'])
        ))
        connection.commit()

    return processed_data_mnn





"""
=====================================
|        Harmmony Features          |
=====================================
"""
#region Inharmonicity extraction
def extract_inharmonicity(mono_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):
    processed_data_inharmonicity = {}

    window = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    spectral_peaks = es.SpectralPeaks(magnitudeThreshold=0.0001, maxPeaks=100)
    inharmonicity_extractor = es.Inharmonicity()

    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        num_frames = (len(audio) - frame_size) // hop_size
        print(f"[{idx}/{total_tracks}] [Inharmonicity] Processing track {track_id} ({duration_sec:.1f}s, ~{num_frames} frames)")

        inharmonicity_values = []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break

            windowed_frame = window(frame)
            spec = spectrum(windowed_frame)
            frequencies, magnitudes = spectral_peaks(spec)
            # Skip frames with no valid peaks or 0 Hz fundamental
            if len(frequencies) == 0 or frequencies[0] <= 0:
                continue
            inh = inharmonicity_extractor(frequencies, magnitudes)
            inharmonicity_values.append(inh)

        # Compute statistics
        if inharmonicity_values:
            mean_inharmonicity = np.mean(inharmonicity_values)
            median_inharmonicity = np.median(inharmonicity_values)
            std_inharmonicity = np.std(inharmonicity_values)
        else:
            mean_inharmonicity = np.nan
            median_inharmonicity = np.nan
            std_inharmonicity = np.nan

        processed_data_inharmonicity[track_id] = {
            'inharmonicity_values': inharmonicity_values,
            'inharmonicity_mean': mean_inharmonicity,
            'inharmonicity_median': median_inharmonicity,
            'inharmonicity_std': std_inharmonicity,
            'inharmonicity_skewness': skew(inharmonicity_values) if inharmonicity_values else np.nan,
            'inharmonicity_kurtosis': kurtosis(inharmonicity_values) if inharmonicity_values else np.nan,
            'inharmonicity_rms': np.sqrt(np.mean(np.square(inharmonicity_values))) if inharmonicity_values else np.nan,
            'inharmonicity_delta': np.mean(np.diff(inharmonicity_values)) if len(inharmonicity_values) > 1 else 0
        
        }
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, inharmonicity_mean, inharmonicity_median, inharmonicity_std, inharmonicity_skewness, inharmonicity_kurtosis, inharmonicity_rms, inharmonicity_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    inharmonicity_mean = excluded.inharmonicity_mean,
                    inharmonicity_median = excluded.inharmonicity_median,
                    inharmonicity_std = excluded.inharmonicity_std,
                    inharmonicity_skewness = excluded.inharmonicity_skewness, 
                    inharmonicity_kurtosis = excluded.inharmonicity_kurtosis , 
                    inharmonicity_rms = excluded.inharmonicity_rms, 
                    inharmonicity_delta = excluded.inharmonicity_delta"""
        # Get the correct values from the processed_data_pitch_melodia dict for this track
        vals = processed_data_inharmonicity[track_id]
        cursor.execute(sql, (
            track_id,
            float(vals['inharmonicity_mean']),
            float(vals['inharmonicity_median']),
            float(vals['inharmonicity_std']),
            float(vals['inharmonicity_skewness']),
            float(vals['inharmonicity_kurtosis']),
            float(vals['inharmonicity_rms']),
            float(vals['inharmonicity_delta'])
        ))
        connection.commit()
    return processed_data_inharmonicity

# Test print
#print(processed_data_inharmonicity)
#endregion

#region Chromagram extraction 


def extract_chromagram(mono_loaded_audio,
                    frame_size=2048,
                    hop_size=1024,
                    ):
    processed_data_chromogram = {}

    window   = es.Windowing(type='hann',   size=frame_size)
    spectrum = es.Spectrum()
    chroma   = es.Chromagram()  

    total = len(mono_loaded_audio)
    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total}] [Chroma] Processing track {track_id}")

        frames = []
        for i in range(0, len(audio)-frame_size, hop_size):
            frame = window(audio[i:i+frame_size])
            spec  = spectrum(frame)            
            spec_padded = pad(spec, (0, 32768 - spec.shape[0]), mode='constant')
            c = chroma(spec_padded)            
            frames.append(c)

        chroma_array = np.vstack(frames) if frames else np.empty((0,12))
        
        # Calculate statistics for each chroma dimension
        if chroma_array.size > 0:
            chroma_mean = np.mean(chroma_array, axis=0)
            chroma_median = np.median(chroma_array, axis=0)
            chroma_std = np.std(chroma_array, axis=0)
            chroma_skewness = skew(chroma_array, axis=0)
            chroma_kurtosis = kurtosis(chroma_array, axis=0)
            chroma_rms = np.sqrt(np.mean(np.square(chroma_array), axis=0))
            chroma_delta = np.mean(np.diff(chroma_array, axis=0), axis=0) if len(chroma_array) > 1 else np.zeros(12)
        else:
            chroma_mean = np.zeros(12)
            chroma_median = np.zeros(12)
            chroma_std = np.zeros(12)
            chroma_skewness = np.zeros(12)
            chroma_kurtosis = np.zeros(12)
            chroma_rms = np.zeros(12)
            chroma_delta = np.zeros(12)

        processed_data_chromogram[track_id] = {
            'chroma_values': chroma_array.tolist(),
            'chroma_mean': chroma_mean.tolist(),
            'chroma_median': chroma_median.tolist(),
            'chroma_std': chroma_std.tolist(),
            'chroma_skewness': chroma_skewness.tolist(),
            'chroma_kurtosis': chroma_kurtosis.tolist(),
            'chroma_rms': chroma_rms.tolist(),
            'chroma_delta': chroma_delta.tolist()
        }
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, chroma_mean, chroma_median, chroma_std, chroma_skewness, chroma_kurtosis, chroma_rms, chroma_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    chroma_mean = excluded.chroma_mean,
                    chroma_median = excluded.chroma_median,
                    chroma_std = excluded.chroma_std,
                    chroma_skewness = excluded.chroma_skewness, 
                    chroma_kurtosis = excluded.chroma_kurtosis , 
                    chroma_rms = excluded.chroma_rms, 
                    chroma_delta = excluded.chroma_delta"""
        # Store only the mean of each chroma statistics array as a float
        cursor.execute(sql, (
            track_id,
            float(np.mean(chroma_mean)),
            float(np.mean(chroma_median)),
            float(np.mean(chroma_std)),
            float(np.mean(chroma_skewness)),
            float(np.mean(chroma_kurtosis)),
            float(np.mean(chroma_rms)),
            float(np.mean(chroma_delta))
        ))
        connection.commit()

    return processed_data_chromogram



# Test print
#print(processed_data_chromogram)

#endregion

#region HPCP Extraction (Harmonic Pitch Class Profile)

def extract_hpcp(mono_loaded_audio,
                frame_size=2048,
                hop_size=1024,
                bins_per_octave=12,
                min_freq=50,
                max_freq=5000):

    processed_data_hpcp = {}

    window       = es.Windowing(type='hann', size=frame_size)
    spectrum     = es.Spectrum()
    spectral_peaks = es.SpectralPeaks(orderBy='magnitude', magnitudeThreshold=0.01)
    hpcp         = es.HPCP(size=bins_per_octave,
                        minFrequency=min_freq,
                        maxFrequency=max_freq,
                        referenceFrequency=440.0)

    total = len(mono_loaded_audio)
    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total}] [HPCP] Processing track {track_id}")
        frames = []
        for i in range(0, len(audio)-frame_size, hop_size):
            frame = window(audio[i:i+frame_size])
            spec  = spectrum(frame)
            freqs, mags = spectral_peaks(spec)
            c = hpcp(freqs, mags)  
            frames.append(c)

        hpcp_array = np.vstack(frames) if frames else np.empty((0, bins_per_octave))
        
        # Calculate statistics for each HPCP bin
        if hpcp_array.size > 0:
            hpcp_mean = np.mean(hpcp_array, axis=0)
            hpcp_median = np.median(hpcp_array, axis=0)
            hpcp_std = np.std(hpcp_array, axis=0)
            hpcp_skewness = skew(hpcp_array, axis=0)
            hpcp_kurtosis = kurtosis(hpcp_array, axis=0)
            hpcp_rms = np.sqrt(np.mean(np.square(hpcp_array), axis=0))
            hpcp_delta = np.mean(np.diff(hpcp_array, axis=0), axis=0) if len(hpcp_array) > 1 else np.zeros(bins_per_octave)
        else:
            hpcp_mean = np.zeros(bins_per_octave)
            hpcp_median = np.zeros(bins_per_octave)
            hpcp_std = np.zeros(bins_per_octave)
            hpcp_skewness = np.zeros(bins_per_octave)
            hpcp_kurtosis = np.zeros(bins_per_octave)
            hpcp_rms = np.zeros(bins_per_octave)
            hpcp_delta = np.zeros(bins_per_octave)

        processed_data_hpcp[track_id] = {
            'hpcp_values': hpcp_array.tolist(),
            'hpcp_mean': hpcp_mean.tolist(),
            'hpcp_median': hpcp_median.tolist(),
            'hpcp_std': hpcp_std.tolist(),
            'hpcp_skewness': hpcp_skewness.tolist(),
            'hpcp_kurtosis': hpcp_kurtosis.tolist(),
            'hpcp_rms': hpcp_rms.tolist(),
            'hpcp_delta': hpcp_delta.tolist()
        }
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, hpcp_mean, hpcp_median, hpcp_std, hpcp_skewness, hpcp_kurtosis, hpcp_rms, hpcp_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    hpcp_mean = excluded.hpcp_mean,
                    hpcp_median = excluded.hpcp_median,
                    hpcp_std = excluded.hpcp_std,
                    hpcp_skewness = excluded.hpcp_skewness, 
                    hpcp_kurtosis = excluded.hpcp_kurtosis , 
                    hpcp_rms = excluded.hpcp_rms, 
                    hpcp_delta = excluded.hpcp_delta"""
        # Get the correct values from the processed_data_pitch_melodia dict for this track
        #vals = processed_data_hpcp[track_id]
        cursor.execute(sql, (
            track_id,
            float(np.mean(hpcp_mean)),
            float(np.mean(hpcp_median)),
            float(np.mean(hpcp_std)),
            float(np.mean(hpcp_skewness)),
            float(np.mean(hpcp_kurtosis)),
            float(np.mean(hpcp_rms)),
            float(np.mean(hpcp_delta))
        ))
        connection.commit()

    return processed_data_hpcp


# Test print
#print(processed_data_hpcp)
#endregion

#region Key Extraction
def extract_key(mono_loaded_audio, sample_rate=44100):
    processed_data_key = {}
    
    key_extractor = es.KeyExtractor(sampleRate=sample_rate)

    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        print(f"[{idx}/{total_tracks}] [Key] Processing track {track_id} ({duration_sec:.1f}s)")

        try:
            key, scale, strength = key_extractor(audio)

            processed_data_key[track_id] = {
                'key': key,
                'scale': scale,
                'strength': strength
            }

        except Exception as e:
            print(f"Error processing track {track_id}: {e}")
            continue
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, key, scale, strength) values (?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    key = excluded.key,
                    scale = excluded.scale,
                    strength = excluded.strength"""
        vals = processed_data_key[track_id]     
        cursor.execute(sql, (
        track_id,
        vals['key'],
        vals['scale'],
        vals['strength']
        ))
        connection.commit()
    return processed_data_key


# Test print
#print(processed_data_key)
#endregion

#region Chord Extraction
def extract_chord_progression_from_hpcp(processed_data_hpcp):

    chord_progression_data = {}
    chords_detector = es.ChordsDetection()


    for idx, (track_id, hpcp_dict) in enumerate(processed_data_hpcp.items(), start=1):
        print(f"[{idx}/{len(processed_data_hpcp)}] [Chord Progression] Processing HPCP for track {track_id}")

        try:
            # Always extract the 'hpcp_values' key, which should be a list of lists
            if isinstance(hpcp_dict, dict) and 'hpcp_values' in hpcp_dict:
                hpcps = hpcp_dict['hpcp_values']
            else:
                hpcps = hpcp_dict

            # Validate shape
            if not isinstance(hpcps, list) or not all(isinstance(v, (list, tuple)) for v in hpcps):
                raise ValueError("HPCP data must be a list of lists (vector<vector<float>>)")

            chord_labels = chords_detector(hpcps)

            chord_progression_data[track_id] = {
                'chord_progression': chord_labels
            }

        except Exception as e:
            print(f"Error processing track {track_id}: {e}")
            continue
        
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, chord_progression) values (?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    chord_progression = excluded.chord_progression"""
        vals = chord_progression_data[track_id]
        # Convert chord_progression (Pitch Class Vectors) to comma-separated string
        chord_vec = vals['chord_progression']
        if isinstance(chord_vec, (list, tuple)):
            chord_vec_str = ','.join(str(x) for x in chord_vec)
        else:
            chord_vec_str = str(chord_vec)
        cursor.execute(sql, (
            track_id,
            chord_vec_str))
        connection.commit()
    return chord_progression_data


# Test print
#print(processed_data_chord_progression)
#endregion

#region Spectral Peaks Extraction
def extract_spectral_peaks(mono_loaded_audio, frame_size=2048, hop_size=1024):
    window = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    spectral_peaks = es.SpectralPeaks(orderBy='frequency')

    peak_data = {}

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{len(mono_loaded_audio)}] [Peaks] Processing track {track_id}")

        freqs_all = []
        mags_all = []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i+frame_size]
            windowed = window(frame)
            spec = spectrum(windowed)
            freqs, mags = spectral_peaks(spec)

            freqs_all.append(freqs)
            mags_all.append(mags)

        if freqs_all and mags_all:
            all_freqs = np.hstack(freqs_all)
            all_mags = np.hstack(mags_all)

            # Sort frequencies and magnitudes together
            sorted_indices = np.argsort(all_freqs)
            sorted_freqs = all_freqs[sorted_indices]
            sorted_mags = all_mags[sorted_indices]

            peak_data[track_id] = (sorted_freqs.tolist(), sorted_mags.tolist())
        else:
            peak_data[track_id] = ([], [])
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, spectral_peaks) values (?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    spectral_peaks = excluded.spectral_peaks"""
        vals = peak_data[track_id]
        # Convert both lists to comma-separated strings and join with a semicolon
        freqs_str = ','.join(str(x) for x in vals[0])
        mags_str = ','.join(str(x) for x in vals[1])
        spectral_peaks_str = f"{freqs_str};{mags_str}"
        cursor.execute(sql, (
            track_id,
            spectral_peaks_str
        ))
        connection.commit()
    return peak_data

# Test print
#print(processed_spectral_peaks)
# Shorting Frequencies and Magnitudes Together for Dissonance extracion


#end region
#region Dissonance Extraction

def extract_dissonance_from_peaks(processed_spectral_peaks):


    dissonance_data = {}
    dissonance_fn = es.Dissonance()

    for idx, (track_id, (frequencies, magnitudes)) in enumerate(processed_spectral_peaks.items(), start=1):
        print(f"[{idx}/{len(processed_spectral_peaks)}] [Dissonance] Processing track {track_id}")

        try:
            # Ensure lists and sorted
            freqs = list(frequencies)
            mags = list(magnitudes)

            if len(freqs) < 2:
                raise ValueError("Not enough spectral peaks for dissonance calculation")

            if not all(freqs[i] <= freqs[i+1] for i in range(len(freqs)-1)):
                raise ValueError("Frequencies must be sorted in ascending order")

            # Call Essentia's dissonance function
            dissonance_value = dissonance_fn(freqs, mags)

            dissonance_data[track_id] = {
                'dissonance': float(dissonance_value)
            }

        except Exception as e:
            print(f"Error processing track {track_id}: {e}")
            continue
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, dissonance) values (?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    dissonance = excluded.dissonance"""
        vals = dissonance_data[track_id]     
        cursor.execute(sql, (
        track_id,
        float(vals['dissonance'])
        ))
        connection.commit()
    return dissonance_data

# Test print
#print(processed_data_dissonance)
#endregion


"""
=====================================
|        Rythm Features          |
=====================================
"""
#region BPM Extraction
def extract_bpm(mono_loaded_audio, sample_rate=44100): 
    
    processed_data_bpm = {}
    rhythm_extractor = es.RhythmExtractor2013(method="multifeature")

    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        print(f"[{idx}/{total_tracks}] [BPM] Processing track {track_id} ({duration_sec:.1f}s)")

        try:
            bpm, ticks, _, _, _ = rhythm_extractor(audio)

            # Calculate standard deviation of instantaneous tempo
            if len(ticks) >= 2:
                iois = np.diff(ticks)  # Inter-Onset Intervals in seconds
                instantaneous_bpms = 60.0 / iois  # Convert to BPM
                tempo_std = float(np.std(instantaneous_bpms)) # Tempo Stability extraction
            else:
                tempo_std = None  # Not enough ticks to compute std

        except Exception as e:
            print(f"Error processing BPM for track {track_id}: {e}")
            bpm = None
            ticks = []
            tempo_std = None

        processed_data_bpm[track_id] = {
            'bpm': bpm,
            'ticks': ticks,
            'tempo_std': tempo_std
        }

    return processed_data_bpm

# Test print
#print(processed_data_bpm)
#endregion

#region Onset Extraction
def extract_onset_rate(mono_loaded_audio, sample_rate=44100):
    processed_data_onset_rate = {}

    onset_rate_algo = es.OnsetRate()

    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        print(f"[{idx}/{total_tracks}] [Onset Rate] Processing track {track_id} ({duration_sec:.1f}s)")

        try:
            onset_rate = onset_rate_algo(audio)
            processed_data_onset_rate[track_id] = {
                'onset_rate': onset_rate
            }
        except Exception as e:
            print(f"Error processing onset rate for track {track_id}: {e}")
            processed_data_onset_rate[track_id] = {
                'onset_rate': None
            }

    return processed_data_onset_rate

# Test print
#print(processed_data_onset_rate)
#endregion

#region Beat Histogram Extraction TODO: fix this
def extract_beat_histogram(mono_loaded_audio, frame_size=1024, hop_size=512, sample_rate=44100):
    processed_data_beat_histogram = {}

    windowing = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    cartesian_to_polar = es.CartesianToPolar()
    onset_detection = es.OnsetDetection(method='complex')
    bpm_histogram_algo = es.BpmHistogram()

    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        duration_sec = len(audio) / sample_rate
        print(f"[{idx}/{total_tracks}] [Beat Histogram] Processing track {track_id} ({duration_sec:.1f}s)")

        novelty_curve = []

        try:
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i + frame_size]
                windowed = windowing(frame)
                spec = spectrum(windowed)
                mag, phase = cartesian_to_polar(spec)
                onset_val = onset_detection(mag, phase)
                novelty_curve.append(float(onset_val))

            if len(novelty_curve) == 0:
                raise ValueError("Novelty curve (onset function) is empty.")

            # Convert to plain Python float list
            novelty_curve_floats = [float(x) for x in novelty_curve]

            histogram, bpm1, bpm2 = bpm_histogram_algo(novelty_curve_floats)

            processed_data_beat_histogram[track_id] = {
                'beat_histogram': histogram,
                'bpm_peak_1': bpm1,
                'bpm_peak_2': bpm2
            }

        except Exception as e:
            print(f"Error processing beat histogram for track {track_id}: {e}")
            processed_data_beat_histogram[track_id] = {
                'beat_histogram': None,
                'bpm_peak_1': None,
                'bpm_peak_2': None
            }

    return processed_data_beat_histogram
# Call the function to extract onset rate

#processed_data_beat_histogram = extract_beat_histogram(mono_loaded_audio) TODO: fix this
#Test print
#print(processed_data_beat_histogram)
#endregion

"""
=====================================
|        Dynamic Features          |
=====================================
"""
#region Loudness Extraction
def extract_loudness(mono_loaded_audio, sample_rate=44100, frame_size=2048, hop_size=1024):

    loudness_algo = es.Loudness()

    processed_loudness = {}
    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total_tracks}] [Loudness] Processing track {track_id}")

        try:
            loudness_values = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i+frame_size]
                loudness_val = loudness_algo(frame)
                loudness_values.append(loudness_val)

            if loudness_values:
                loudness_array = np.array(loudness_values)
                processed_loudness[track_id] = {
                    'loudness_values': loudness_values,
                    'loudness_mean': float(np.mean(loudness_array)),
                    'loudness_median': float(np.median(loudness_array)),
                    'loudness_std': float(np.std(loudness_array)),
                    'loudness_skewness': float(skew(loudness_array)),
                    'loudness_kurtosis': float(kurtosis(loudness_array)),
                    'loudness_rms': float(np.sqrt(np.mean(np.square(loudness_array)))),
                    'loudness_delta': float(np.mean(np.diff(loudness_array))) if len(loudness_array) > 1 else 0.0
                }
            else:
                processed_loudness[track_id] = {
                    'loudness_values': [],
                    'loudness_mean': 0.0,
                    'loudness_median': 0.0,
                    'loudness_std': 0.0,
                    'loudness_skewness': 0.0,
                    'loudness_kurtosis': 0.0,
                    'loudness_rms': 0.0,
                    'loudness_delta': 0.0
                }
            cursor = connection.cursor()
            sql = """INSERT INTO processed_sound_data (
                        track_id, loudness_mean, loudness_median, loudness_std, loudness_skewness, loudness_kurtosis, loudness_rms, loudness_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                        ON CONFLICT(track_id) DO UPDATE SET 
                        loudness_mean = excluded.loudness_mean,
                        loudness_median = excluded.loudness_median,
                        loudness_std = excluded.loudness_std,
                        loudness_skewness = excluded.loudness_skewness, 
                        loudness_kurtosis = excluded.loudness_kurtosis , 
                        loudness_rms = excluded.loudness_rms, 
                        loudness_delta = excluded.loudness_delta"""
            # Get the correct values from the processed_data_pitch_melodia dict for this track
            vals = processed_loudness[track_id]
            cursor.execute(sql, (
            track_id,
            float(vals['loudness_mean']),
            float(vals['loudness_median']),
            float(vals['loudness_std']),
            float(vals['loudness_skewness']),
            float(vals['loudness_kurtosis']),
            float(vals['loudness_rms']),
            float(vals['loudness_delta'])
        ))
            connection.commit()

        except Exception as e:
            print(f"Error processing loudness for track {track_id}: {e}")
            processed_loudness[track_id] = None

    return processed_loudness

# Test print
#print(processed_loudness_mean)
#endregion

#region Dynamic Range Extraction
def extract_dynamic_range(mono_loaded_audio, frame_size=2048, hop_size=1024):

    rms_algo = es.RMS()

    processed_dynamic_range = {}
    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total_tracks}] [Dynamic Range] Processing track {track_id}")

        try:
            rms_values = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i+frame_size]
                rms_val = rms_algo(frame)
                rms_values.append(rms_val)

            if rms_values:
                rms_array = np.array(rms_values)
                processed_dynamic_range[track_id] = {
                    'dynamic_values': rms_values,
                    'dynamic_mean': float(np.mean(rms_array)),
                    'dynamic_median': float(np.median(rms_array)),
                    'dynamic_std': float(np.std(rms_array)),
                    'dynamic_skewness': float(skew(rms_array)),
                    'dynamic_kurtosis': float(kurtosis(rms_array)),
                    'dynamic_rms': float(np.sqrt(np.mean(np.square(rms_array)))),
                    'dynamic_delta': float(np.mean(np.diff(rms_array))) if len(rms_array) > 1 else 0.0,
                    'dynamic_range': float(max(rms_values) - min(rms_values))
                }
            else:
                processed_dynamic_range[track_id] = {
                    'dynamic_values': [],
                    'dynamic_mean': 0.0,
                    'dynamic_median': 0.0,
                    'dynamic_std': 0.0,
                    'dynamic_skewness': 0.0,
                    'dynamic_kurtosis': 0.0,
                    'dynamic_rms': 0.0,
                    'dynamic_delta': 0.0,
                    'dynamic_range': 0.0
                }
            cursor = connection.cursor()
            sql = """INSERT INTO processed_sound_data (
                        track_id, dynamic_mean, dynamic_median, dynamic_std, dynamic_skewness, dynamic_kurtosis, dynamic_rms, dynamic_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                        ON CONFLICT(track_id) DO UPDATE SET 
                        dynamic_mean = excluded.dynamic_mean,
                        dynamic_median = excluded.dynamic_median,
                        dynamic_std = excluded.dynamic_std,
                        dynamic_skewness = excluded.dynamic_skewness, 
                        dynamic_kurtosis = excluded.dynamic_kurtosis , 
                        dynamic_rms = excluded.dynamic_rms, 
                        dynamic_delta = excluded.dynamic_delta"""
            # Get the correct values from the processed_data_pitch_melodia dict for this track
            vals = processed_dynamic_range[track_id]
            cursor.execute(sql, (
            track_id, 
            float(vals['dynamic_mean']),
            float(vals['dynamic_median']),
            float(vals['dynamic_std']),
            float(vals['dynamic_skewness']),
            float(vals['dynamic_kurtosis']),
            float(vals['dynamic_rms']),
            float(vals['dynamic_delta'])
            ))
            connection.commit()

        except Exception as e:
            print(f"Error processing dynamic range for track {track_id}: {e}")
            processed_dynamic_range[track_id] = None

    return processed_dynamic_range

# Test print
#print(processed_dynamic_range)
#endregion

#region RMS Energy STD Extraction
def extract_rms_energy_std(mono_loaded_audio, frame_size=2048, hop_size=1024):

    rms_algo = es.RMS()

    processed_rms_std = {}
    total_tracks = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total_tracks}] [RMS Energy] Processing track {track_id}")

        try:
            rms_values = []
            for i in range(0, len(audio) - frame_size, hop_size):
                frame = audio[i:i+frame_size]
                rms_val = rms_algo(frame)
                rms_values.append(rms_val)

            if rms_values:
                rms_array = np.array(rms_values)
                processed_rms_std[track_id] = {
                    'rms_values': rms_values,
                    'rms_mean': float(np.mean(rms_array)),
                    'rms_median': float(np.median(rms_array)),
                    'rms_std': float(np.std(rms_array)),
                    'rms_skewness': float(skew(rms_array)),
                    'rms_kurtosis': float(kurtosis(rms_array)),
                    'rms_rms': float(np.sqrt(np.mean(np.square(rms_array)))),
                    'rms_delta': float(np.mean(np.diff(rms_array))) if len(rms_array) > 1 else 0.0
                }
            else:
                processed_rms_std[track_id] = {
                    'rms_values': [],
                    'rms_mean': 0.0,
                    'rms_median': 0.0,
                    'rms_std': 0.0,
                    'rms_skewness': 0.0,
                    'rms_kurtosis': 0.0,
                    'rms_rms': 0.0,
                    'rms_delta': 0.0
                }
            cursor = connection.cursor()
            sql = """INSERT INTO processed_sound_data (
                        track_id, rms_mean, rms_median, rms_std, rms_skewness, rms_kurtosis, rms_rms, rms_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                        ON CONFLICT(track_id) DO UPDATE SET 
                        rms_mean = excluded.rms_mean,
                        rms_median = excluded.rms_median,
                        rms_std = excluded.rms_std,
                        rms_skewness = excluded.rms_skewness, 
                        rms_kurtosis = excluded.rms_kurtosis , 
                        rms_rms = excluded.rms_rms, 
                        rms_delta = excluded.rms_delta"""
            # Get the correct values from the processed_data_pitch_melodia dict for this track
            vals = processed_rms_std[track_id]
            cursor.execute(sql, (
            track_id, 
            float(vals['rms_mean']),
            float(vals['rms_median']),
            float(vals['rms_std']),
            float(vals['rms_skewness']),
            float(vals['rms_kurtosis']),
            float(vals['rms_rms']),
            float(vals['rms_delta'])
            ))
            connection.commit()

        except Exception as e:
            print(f"Error processing RMS std for track {track_id}: {e}")
            processed_rms_std[track_id] = None

    return processed_rms_std

# Test print
#print(processed_rms_energy_std)
#endregion


"""
=====================================
|        Tone Colour Features          |
=====================================
"""
#region MFCC Extraction
def extract_mfcc(mono_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):

    processed_data_mfcc = {}

    window = es.Windowing(type='hann')
    spectrum_algo = es.Spectrum()
    mfcc = es.MFCC(inputSize=frame_size // 2 + 1)

    total_tracks_len = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        mfcc_coeffs = []
        mfcc_bands = []
        times = []

        duration_sec = len(audio) / sample_rate
        num_frames = (len(audio) - frame_size) // hop_size
        print(f"[{idx}/{total_tracks_len}] [MFCC] Processing track {track_id} ({duration_sec:.1f}s, ~{num_frames} frames)")

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break

            spectrum = spectrum_algo(window(frame))
            bands, mfccs = mfcc(spectrum)
            mfcc_bands.append(bands)
            mfcc_coeffs.append(mfccs)
            times.append(i / sample_rate)

            if i % (sample_rate * 10) == 0 and i != 0:
                print(f"    ↳ Processed {i / sample_rate:.1f}s")

        # Calculate mean and std of MFCCs
        if mfcc_coeffs:
            mfcc_array = np.array(mfcc_coeffs)
            processed_data_mfcc[track_id] = {
                'mfcc_coefficients': mfcc_coeffs,
                'mfcc_bands': mfcc_bands,
                'mfcc_times': times,
                'mfcc_mean': mfcc_array.mean(axis=0).tolist(),
                'mfcc_median': np.median(mfcc_array, axis=0).tolist(),
                'mfcc_std': mfcc_array.std(axis=0).tolist(),
                'mfcc_skewness': skew(mfcc_array, axis=0).tolist(),
                'mfcc_kurtosis': kurtosis(mfcc_array, axis=0).tolist(),
                'mfcc_rms': np.sqrt(np.mean(np.square(mfcc_array), axis=0)).tolist(),
                'mfcc_delta': np.mean(np.diff(mfcc_array, axis=0), axis=0).tolist() if len(mfcc_array) > 1 else np.zeros(mfcc_array.shape[1]).tolist()
            }
        else:
            empty_features = []
            processed_data_mfcc[track_id] = {
                'mfcc_coefficients': [],
                'mfcc_bands': [],
                'mfcc_times': [],
                'mfcc_mean': empty_features,
                'mfcc_median': empty_features,
                'mfcc_std': empty_features,
                'mfcc_skewness': empty_features,
                'mfcc_kurtosis': empty_features,
                'mfcc_rms': empty_features,
                'mfcc_delta': empty_features
            }

    return processed_data_mfcc


# Test print
#print(processed_data_mfcc)
#endregion

#region Spectral Centroid Extraction
def extract_spectral_centroid(mono_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):

    centroid_processed_data = {}

    window = es.Windowing(type='hann')
    spectrum_algo = es.Spectrum()
    spectral_centroid = es.CentralMoments()

    total_tracks_len = len(mono_loaded_audio)

    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total_tracks_len}] [Spectral Centroid] Processing track {track_id}")

        centroids = []
        times = []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break

            spectrum = spectrum_algo(window(frame))
            moments = spectral_centroid(spectrum)
            centroid = moments[1]  # 1st moment is the centroid
            centroids.append(centroid)
            times.append(i / sample_rate)

        if centroids:
            centroid_array = np.array(centroids)
            centroid_processed_data[track_id] = {
                'centroid_values': centroids,
                'centroid_times': times,
                'centroid_mean': float(np.mean(centroid_array)),
                'centroid_median': float(np.median(centroid_array)),
                'centroid_std': float(np.std(centroid_array)),
                'centroid_skewness': float(skew(centroid_array)),
                'centroid_kurtosis': float(kurtosis(centroid_array)),
                'centroid_rms': float(np.sqrt(np.mean(np.square(centroid_array)))),
                'centroid_delta': float(np.mean(np.diff(centroid_array))) if len(centroid_array) > 1 else 0.0
            }
        else:
            centroid_processed_data[track_id] = {
                'centroid_values': [],
                'centroid_times': [],
                'centroid_mean': 0.0,
                'centroid_median': 0.0,
                'centroid_std': 0.0,
                'centroid_skewness': 0.0,
                'centroid_kurtosis': 0.0,
                'centroid_rms': 0.0,
                'centroid_delta': 0.0
            }
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, centroid_mean, centroid_median, centroid_std, centroid_skewness, centroid_kurtosis, centroid_rms, centroid_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    centroid_mean = excluded.centroid_mean,
                    centroid_median = excluded.centroid_median,
                    centroid_std = excluded.centroid_std,
                    centroid_skewness = excluded.centroid_skewness, 
                    centroid_kurtosis = excluded.centroid_kurtosis , 
                    centroid_rms = excluded.centroid_rms, 
                    centroid_delta = excluded.centroid_delta"""
        # Get the correct values from the processed_data_pitch_melodia dict for this track
        vals = centroid_processed_data[track_id]
        cursor.execute(sql, (
        track_id, 
        float(vals['centroid_mean']),
        float(vals['centroid_median']),
        float(vals['centroid_std']),
        float(vals['centroid_skewness']),
        float(vals['centroid_kurtosis']),
        float(vals['centroid_rms']),
        float(vals['centroid_delta'])
        ))
        connection.commit()

    return centroid_processed_data

# Test print
#print(processed_data_spectral_centroid)

#endregion
"""
=====================================
|        Form Features          |
=====================================
"""
#region Segment Count Extraction

def extract_segment_boundaries_and_novelty(eqloud_loaded_audio, frame_size=1024, hop_size=512, sample_rate=44100):

    segment_data = {}

    window = es.Windowing(type='hann')
    fft = es.FFT()
    onset_detection = es.OnsetDetection(method='complex')

    total_tracks_len = len(eqloud_loaded_audio)
    for idx, (track_id, audio) in enumerate(eqloud_loaded_audio.items(), start=1):
        print(f"[{idx}/{total_tracks_len}] [Boundaries & Novelty] Processing track {track_id}")

        onset_env = []
        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            w = window(frame)

            fft_complex = fft(w)
            mag = np.abs(fft_complex)
            phase = np.angle(fft_complex)

            onset_val = onset_detection(mag, phase)
            onset_env.append(onset_val)

        onset_env = np.array(onset_env)
        peaks, _ = find_peaks(onset_env, height=0.3, distance=10)
        segment_boundaries = [p * hop_size / sample_rate for p in peaks]

        audio_duration = len(audio) / sample_rate

        segment_data[track_id] = {
            'segment_boundaries_sec': segment_boundaries,
            'onset_envelope': onset_env,
            'audio_duration_sec': audio_duration
        }
        

    return segment_data

# Test print
#print(processed_data_segment_count)
#endregion

#region Segment Duration Extraction
def extract_segment_durations_stats(segment_data, eqloud_loaded_audio):

    durations_stats = {}
    total_tracks_len = len(eqloud_loaded_audio)
    for idx, (track_id, data) in enumerate(segment_data.items(), start=1):
        boundaries = data['segment_boundaries_sec']
        audio_length = data['audio_duration_sec']
        print(f"[{idx}/{total_tracks_len}] [Segment Duration] Processing track {track_id}")

        # include start and end boundaries
        all_boundaries = [0] + boundaries + [audio_length]

        # calculate segment durations
        durations = np.diff(all_boundaries)  # array of durations

        if len(durations) > 0:
            durations_stats[track_id] = {
                'segment_durations': durations.tolist(),
                'segment_mean': float(np.mean(durations)),
                'segment_median': float(np.median(durations)),
                'segment_std': float(np.std(durations)),
                'segment_skewness': float(skew(durations)),
                'segment_kurtosis': float(kurtosis(durations)),
                'segment_rms': float(np.sqrt(np.mean(np.square(durations)))),
                'segment_delta': float(np.mean(np.diff(durations))) if len(durations) > 1 else 0.0
            }
        else:
            durations_stats[track_id] = {
                'segment_durations': [],
                'segment_mean': 0.0,
                'segment_median': 0.0,
                'segment_std': 0.0,
                'segment_skewness': 0.0,
                'segment_kurtosis': 0.0,
                'segment_rms': 0.0,
                'segment_delta': 0.0
            }
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, segment_mean, segment_median, segment_std, segment_skewness, segment_kurtosis, segment_rms, segment_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    segment_mean = excluded.segment_mean,
                    segment_median = excluded.segment_median,
                    segment_std = excluded.segment_std,
                    segment_skewness = excluded.segment_skewness, 
                    segment_kurtosis = excluded.segment_kurtosis , 
                    segment_rms = excluded.segment_rms, 
                    segment_delta = excluded.segment_delta"""
        vals = durations_stats[track_id]        
        cursor.execute(sql, (
        track_id, 
        float(vals['segment_mean']),
        float(vals['segment_median']),
        float(vals['segment_std']),
        float(vals['segment_skewness']),
        float(vals['segment_kurtosis']),
        float(vals['segment_rms']),
        float(vals['segment_delta'])
        ))
        connection.commit()

    return durations_stats

# Test print
#print(processed_data_segment_durations)
#endregion

#region Novelty Curve Extraction
def extract_novelty_stats(segment_data,eqloud_loaded_audio): #TODO: logika thelei eq

    novelty_stats = {}
    total_tracks_len = len(eqloud_loaded_audio)
    for idx, (track_id, data) in enumerate(segment_data.items(), start=1):
        print(f"[{idx}/{total_tracks_len}] [Novelty Curve] Processing track {track_id}")
        onset_env = data['onset_envelope']
        novelty_stats[track_id] = {
            'onset_envelope': onset_env.tolist(),
            'novelty_mean': float(np.mean(onset_env)),
            'novelty_median': float(np.median(onset_env)),
            'novelty_std': float(np.std(onset_env)),
            'novelty_skewness': float(skew(onset_env)),
            'novelty_kurtosis': float(kurtosis(onset_env)),
            'novelty_rms': float(np.sqrt(np.mean(np.square(onset_env)))),
            'novelty_delta': float(np.mean(np.diff(onset_env))) if len(onset_env) > 1 else 0.0
        }
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, novelty_mean, novelty_median, novelty_std, novelty_skewness, novelty_kurtosis, novelty_rms, novelty_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    novelty_mean = excluded.novelty_mean,
                    novelty_median = excluded.novelty_median,
                    novelty_std = excluded.novelty_std,
                    novelty_skewness = excluded.novelty_skewness, 
                    novelty_kurtosis = excluded.novelty_kurtosis , 
                    novelty_rms = excluded.novelty_rms, 
                    novelty_delta = excluded.novelty_delta"""
        vals = novelty_stats[track_id]        
        cursor.execute(sql, (
        track_id, 
        float(vals['novelty_mean']),
        float(vals['novelty_median']),
        float(vals['novelty_std']),
        float(vals['novelty_skewness']),
        float(vals['novelty_kurtosis']),
        float(vals['novelty_rms']),
        float(vals['novelty_delta'])
        ))
        connection.commit()

    return novelty_stats

# Test print
#print(processed_data_novelty_stats)
#endregion

"""
=====================================
|        Expressivity Features          |
=====================================
"""
#region Log Attack Time Extraction
def extract_log_attack_time(mono_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):


    lat_processed_data = {}

    window = es.Windowing(type='hann')
    lat = es.LogAttackTime()

    total_tracks_len = len(mono_loaded_audio)
    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        lat_values = []
        times = []

        duration_sec = len(audio) / sample_rate
        num_frames = (len(audio) - frame_size) // hop_size
        print(f"[{idx}/{total_tracks_len}] [Log Attack Time] Processing track {track_id} ({duration_sec:.1f}s, ~{num_frames} frames)")

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break

            # Apply windowing
            windowed_frame = window(frame)

            # Compute Log Attack Time
            lat_value = lat(windowed_frame)
            lat_values.append(lat_value)
            times.append(i / sample_rate)

            # Optional: log progress every 10 seconds (if needed)
            if i % (sample_rate * 10) == 0 and i != 0:
                print(f"    ↳ Processed {i / sample_rate:.1f}s")

        if lat_values:
            lat_array = np.array(lat_values).flatten()
            lat_skew = float(skew(lat_array)) if lat_array.size > 0 else 0.0
            lat_kurt = float(kurtosis(lat_array)) if lat_array.size > 0 else 0.0
            lat_rms = float(np.sqrt(np.mean(np.square(lat_array)))) if lat_array.size > 0 else 0.0
            lat_delta = float(np.mean(np.diff(lat_array))) if lat_array.size > 1 else 0.0
            lat_processed_data[track_id] = {
                'log_attack_time_values': lat_array.tolist(),
                'log_attack_time_times': times,
                'lat_mean': float(np.mean(lat_array)) if lat_array.size > 0 else 0.0,
                'lat_median': float(np.median(lat_array)) if lat_array.size > 0 else 0.0,
                'lat_std': float(np.std(lat_array)) if lat_array.size > 0 else 0.0,
                'lat_skewness': lat_skew,
                'lat_kurtosis': lat_kurt,
                'lat_rms': lat_rms,
                'lat_delta': lat_delta
            }
        else:
            lat_processed_data[track_id] = {
                'log_attack_time_values': [],
                'log_attack_time_times': [],
                'lat_mean': 0.0,
                'lat_median': 0.0,
                'lat_std': 0.0,
                'lat_skewness': 0.0,
                'lat_kurtosis': 0.0,
                'lat_rms': 0.0,
                'lat_delta': 0.0
            }
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, lat_mean, lat_median, lat_std, lat_skewness, lat_kurtosis, lat_rms, lat_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    lat_mean = excluded.lat_mean,
                    lat_median = excluded.lat_median,
                    lat_std = excluded.lat_std,
                    lat_skewness = excluded.lat_skewness, 
                    lat_kurtosis = excluded.lat_kurtosis , 
                    lat_rms = excluded.lat_rms, 
                    lat_delta = excluded.lat_delta"""
        vals = lat_processed_data[track_id]        
        cursor.execute(sql, (
        track_id, 
        float(vals['lat_mean']),
        float(vals['lat_median']),
        float(vals['lat_std']),
        float(vals['lat_skewness']),
        float(vals['lat_kurtosis']),
        float(vals['lat_rms']),
        float(vals['lat_delta'])
        ))
        connection.commit()

    return lat_processed_data

# Test print
#print(processed_data_log_attack_time)
#endregion

#region Vibrato Extraction
def extract_vibrato(mono_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):


    vibrato_data = {}

    window = es.Windowing(type='hann')
    vibrato = es.Vibrato()

    total_tracks_len = len(mono_loaded_audio)
    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        vibrato_values = []
        times = []

        duration_sec = len(audio) / sample_rate
        num_frames = (len(audio) - frame_size) // hop_size
        print(f"[{idx}/{total_tracks_len}] [Vibrato] Processing track {track_id} ({duration_sec:.1f}s, ~{num_frames} frames)")

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break

            # Apply windowing
            windowed_frame = window(frame)

            # Compute Vibrato 
            vib_value = vibrato(windowed_frame)
            vibrato_values.append(vib_value)
            times.append(i / sample_rate)

            # Optional: log progress every 10 seconds (can be disabled if needed)
            if i % (sample_rate * 10) == 0 and i != 0:
                print(f"    ↳ Processed {i / sample_rate:.1f}s")

        if vibrato_values:
            vibrato_array = np.array(vibrato_values)
            vibrato_data[track_id] = {
                'vibrato_values': vibrato_values,
                'vibrato_times': times,
                'vibrato_mean': float(np.mean(vibrato_array)),
                'vibrato_median': float(np.median(vibrato_array)),
                'vibrato_std': float(np.std(vibrato_array)),
                'vibrato_skewness': float(skew(vibrato_array)),
                'vibrato_kurtosis': float(kurtosis(vibrato_array)),
                'vibrato_rms': float(np.sqrt(np.mean(np.square(vibrato_array)))),
                'vibrato_delta': float(np.mean(np.diff(vibrato_array))) if len(vibrato_array) > 1 else 0.0
            }
        else:
            vibrato_data[track_id] = {
                'vibrato_values': [],
                'vibrato_times': [],
                'vibrato_mean': 0.0,
                'vibrato_median': 0.0,
                'vibrato_std': 0.0,
                'vibrato_skewness': 0.0,
                'vibrato_kurtosis': 0.0,
                'vibrato_rms': 0.0,
                'vibrato_delta': 0.0
            }
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, vibrato_mean, vibrato_median, vibrato_std, vibrato_skewness, vibrato_kurtosis, vibrato_rms, vibrato_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    vibrato_mean = excluded.vibrato_mean,
                    vibrato_median = excluded.vibrato_median,
                    vibrato_std = excluded.vibrato_std,
                    vibrato_skewness = excluded.vibrato_skewness, 
                    vibrato_kurtosis = excluded.vibrato_kurtosis , 
                    vibrato_rms = excluded.vibrato_rms, 
                    vibrato_delta = excluded.vibrato_delta"""
        vals = vibrato_data[track_id]        
        cursor.execute(sql, (
        track_id, 
        float(vals['vibrato_mean']),
        float(vals['vibrato_median']),
        float(vals['vibrato_std']),
        float(vals['vibrato_skewness']),
        float(vals['vibrato_kurtosis']),
        float(vals['vibrato_rms']),
        float(vals['vibrato_delta'])
        ))
        connection.commit()

    return vibrato_data
# Call the function to extract Vibrato Presence
#processed_data_vibrato = extract_vibrato(mono_loaded_audio)
# Test print
#print(processed_data_vibrato)
#endregion


"""
=====================================
|        Texture Features          |
=====================================
"""
#region Spectral Flatness Extraction
def extract_spectral_flatness(mono_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):

    flatness_data = {}

    total_tracks = len(mono_loaded_audio)
    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total_tracks}] [Spectral Flatness] Processing track {track_id}")

        try:
            # Convert to numpy array (Librosa expects float32)
            audio_np = np.array(audio).astype(np.float32)
            S_flat = librosa.feature.spectral_flatness(y=audio_np, n_fft=frame_size, hop_length=hop_size)
            S_flat = S_flat.squeeze()
            flatness_data[track_id] = {
                'flatness_values': S_flat.tolist(),
                'flatness_mean': float(np.mean(S_flat)),
                'flatness_median': float(np.median(S_flat)),
                'flatness_std': float(np.std(S_flat)),
                'flatness_skewness': float(skew(S_flat)),
                'flatness_kurtosis': float(kurtosis(S_flat)),
                'flatness_rms': float(np.sqrt(np.mean(np.square(S_flat)))),
                'flatness_delta': float(np.mean(np.diff(S_flat))) if len(S_flat) > 1 else 0.0
            }

        except Exception as e:
            print(f"Error processing spectral flatness for track {track_id}: {e}")
            flatness_data[track_id] = {
                'flatness_values': [],
                'flatness_mean': 0.0,
                'flatness_median': 0.0,
                'flatness_std': 0.0,
                'flatness_skewness': 0.0,
                'flatness_kurtosis': 0.0,
                'flatness_rms': 0.0,
                'flatness_delta': 0.0
            }
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, flatness_mean, flatness_median, flatness_std, flatness_skewness, flatness_kurtosis, flatness_rms, flatness_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    flatness_mean = excluded.flatness_mean,
                    flatness_median = excluded.flatness_median,
                    flatness_std = excluded.flatness_std,
                    flatness_skewness = excluded.flatness_skewness, 
                    flatness_kurtosis = excluded.flatness_kurtosis , 
                    flatness_rms = excluded.flatness_rms, 
                    flatness_delta = excluded.flatness_delta"""
        vals = flatness_data[track_id]        
        cursor.execute(sql, (
        track_id, 
        float(vals['flatness_mean']),
        float(vals['flatness_median']),
        float(vals['flatness_std']),
        float(vals['flatness_skewness']),
        float(vals['flatness_kurtosis']),
        float(vals['flatness_rms']),
        float(vals['flatness_delta'])
        ))
        connection.commit()

    return flatness_data

# Test print
#print(processed_data_spectral_flatness)
#endregion


#region Tristimulus Extraction
def extract_tristimulus(mono_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):

    tristimulus_data = {}

    window = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    hpcp = es.HarmonicPeaks()
    trist = es.Tristimulus()

    total_tracks = len(mono_loaded_audio)
    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total_tracks}] [Tristimulus] Processing track {track_id}")

        t1_vals, t2_vals, t3_vals = [], [], []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break

            spec = spectrum(window(frame))
            freq, mag = es.SpectralPeaks()(spec)

            if len(freq) >= 3:  # Needs at least 3 partials
                try:
                    t1, t2, t3 = trist(freq, mag)
                    t1_vals.append(t1)
                    t2_vals.append(t2)
                    t3_vals.append(t3)
                except:
                    continue

        if t1_vals:
            t1_array = np.array(t1_vals)
            t2_array = np.array(t2_vals)
            t3_array = np.array(t3_vals)
            tristimulus_data[track_id] = {
                't1_values': t1_vals,
                't2_values': t2_vals,
                't3_values': t3_vals,
                't1_mean': float(np.mean(t1_array)),
                't1_median': float(np.median(t1_array)),
                't1_std': float(np.std(t1_array)),
                't1_skewness': float(skew(t1_array)),
                't1_kurtosis': float(kurtosis(t1_array)),
                't1_rms': float(np.sqrt(np.mean(np.square(t1_array)))),
                't1_delta': float(np.mean(np.diff(t1_array))) if len(t1_array) > 1 else 0.0,
                't2_mean': float(np.mean(t2_array)),
                't2_median': float(np.median(t2_array)),
                't2_std': float(np.std(t2_array)),
                't2_skewness': float(skew(t2_array)),
                't2_kurtosis': float(kurtosis(t2_array)),
                't2_rms': float(np.sqrt(np.mean(np.square(t2_array)))),
                't2_delta': float(np.mean(np.diff(t2_array))) if len(t2_array) > 1 else 0.0,
                't3_mean': float(np.mean(t3_array)),
                't3_median': float(np.median(t3_array)),
                't3_std': float(np.std(t3_array)),
                't3_skewness': float(skew(t3_array)),
                't3_kurtosis': float(kurtosis(t3_array)),
                't3_rms': float(np.sqrt(np.mean(np.square(t3_array)))),
                't3_delta': float(np.mean(np.diff(t3_array))) if len(t3_array) > 1 else 0.0
            }
        else:
            tristimulus_data[track_id] = {
                't1_values': [], 't2_values': [], 't3_values': [],
                't1_mean': 0.0, 't1_median': 0.0, 't1_std': 0.0,
                't1_skewness': 0.0, 't1_kurtosis': 0.0, 't1_rms': 0.0, 't1_delta': 0.0,
                't2_mean': 0.0, 't2_median': 0.0, 't2_std': 0.0,
                't2_skewness': 0.0, 't2_kurtosis': 0.0, 't2_rms': 0.0, 't2_delta': 0.0,
                't3_mean': 0.0, 't3_median': 0.0, 't3_std': 0.0,
                't3_skewness': 0.0, 't3_kurtosis': 0.0, 't3_rms': 0.0, 't3_delta': 0.0
            }
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, t1_mean, t1_median, t1_std, t1_skewness, t1_kurtosis, t1_rms, t1_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    t1_mean = excluded.t1_mean,
                    t1_median = excluded.t1_median,
                    t1_std = excluded.t1_std,
                    t1_skewness = excluded.t1_skewness, 
                    t1_kurtosis = excluded.t1_kurtosis , 
                    t1_rms = excluded.t1_rms, 
                    t1_delta = excluded.t1_delta"""
        vals = tristimulus_data[track_id]  
        cursor.execute(sql, (
        track_id, 
        float(vals['t1_mean']),
        float(vals['t1_median']),
        float(vals['t1_std']),
        float(vals['t1_skewness']),
        float(vals['t1_kurtosis']),
        float(vals['t1_rms']),
        float(vals['t1_delta'])
        ))
        connection.commit()


        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, t2_mean, t2_median, t2_std, t2_skewness, t2_kurtosis, t2_rms, t2_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    t2_mean = excluded.t2_mean,
                    t2_median = excluded.t2_median,
                    t2_std = excluded.t2_std,
                    t2_skewness = excluded.t2_skewness, 
                    t2_kurtosis = excluded.t2_kurtosis , 
                    t2_rms = excluded.t2_rms, 
                    t2_delta = excluded.t2_delta"""
        vals = tristimulus_data[track_id]        
        cursor.execute(sql, (
        track_id, 
        float(vals['t2_mean']),
        float(vals['t2_median']),
        float(vals['t2_std']),
        float(vals['t2_skewness']),
        float(vals['t2_kurtosis']),
        float(vals['t2_rms']),
        float(vals['t2_delta'])
        ))
        connection.commit()


        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, t3_mean, t3_median, t3_std, t3_skewness, t3_kurtosis, t3_rms, t3_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    t3_mean = excluded.t3_mean,
                    t3_median = excluded.t3_median,
                    t3_std = excluded.t3_std,
                    t3_skewness = excluded.t3_skewness, 
                    t3_kurtosis = excluded.t3_kurtosis , 
                    t3_rms = excluded.t3_rms, 
                    t3_delta = excluded.t3_delta"""
        vals = tristimulus_data[track_id]        
        cursor.execute(sql, (
        track_id, 
        float(vals['t3_mean']),
        float(vals['t3_median']),
        float(vals['t3_std']),
        float(vals['t3_skewness']),
        float(vals['t3_kurtosis']),
        float(vals['t3_rms']),
        float(vals['t3_delta'])
        ))
        connection.commit()

    return tristimulus_data

# Test print
#print(processed_data_tristimulus)
#endregion


#region Odd/Even Harmonic Energy Ratio Extraction
def extract_odd_even_harmonic_ratio(mono_loaded_audio, frame_size=2048, hop_size=1024):

    harmonic_ratio_data = {}

    window = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    spectral_peaks = es.SpectralPeaks(orderBy='frequency')
    odd_even_ratio = es.OddToEvenHarmonicEnergyRatio()

    total_tracks = len(mono_loaded_audio)
    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total_tracks}] [Odd Even Harmonics] Processing track {track_id}")

        ratios = []

        for i in range(0, len(audio) - frame_size, hop_size):
            frame = audio[i:i + frame_size]
            if len(frame) < frame_size:
                break

            try:
                spec = spectrum(window(frame))
                freqs, mags = spectral_peaks(spec)

                if len(freqs) >= 2:
                    ratio = odd_even_ratio(freqs, mags)
                    ratios.append(ratio)
            except:
                continue

        if ratios:
            ratios_array = np.array(ratios)
            harmonic_ratio_data[track_id] = {
                'harmonic_ratio_values': ratios,
                'harmonic_ratio_mean': float(np.mean(ratios_array)),
                'harmonic_ratio_median': float(np.median(ratios_array)),
                'harmonic_ratio_std': float(np.std(ratios_array)),
                'harmonic_ratio_skewness': float(skew(ratios_array)),
                'harmonic_ratio_kurtosis': float(kurtosis(ratios_array)),
                'harmonic_ratio_rms': float(np.sqrt(np.mean(np.square(ratios_array)))),
                'harmonic_ratio_delta': float(np.mean(np.diff(ratios_array))) if len(ratios_array) > 1 else 0.0
            }
        else:
            harmonic_ratio_data[track_id] = {
                'harmonic_ratio_values': [],
                'harmonic_ratio_mean': 0.0,
                'harmonic_ratio_median': 0.0,
                'harmonic_ratio_std': 0.0,
                'harmonic_ratio_skewness': 0.0,
                'harmonic_ratio_kurtosis': 0.0,
                'harmonic_ratio_rms': 0.0,
                'harmonic_ratio_delta': 0.0
            }

        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, harmonic_ratio_mean, harmonic_ratio_median, harmonic_ratio_std, harmonic_ratio_skewness, harmonic_ratio_kurtosis, harmonic_ratio_rms, harmonic_ratio_delta) values (?, ?, ?, ?, ?, ?, ?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    harmonic_ratio_mean = excluded.harmonic_ratio_mean,
                    harmonic_ratio_median = excluded.harmonic_ratio_median,
                    harmonic_ratio_std = excluded.harmonic_ratio_std,
                    harmonic_ratio_skewness = excluded.harmonic_ratio_skewness, 
                    harmonic_ratio_kurtosis = excluded.harmonic_ratio_kurtosis , 
                    harmonic_ratio_rms = excluded.harmonic_ratio_rms, 
                    harmonic_ratio_delta = excluded.harmonic_ratio_delta"""
        vals = harmonic_ratio_data[track_id]        
        cursor.execute(sql, (
        track_id, 
        float(vals['harmonic_ratio_mean']),
        float(vals['harmonic_ratio_median']),
        float(vals['harmonic_ratio_std']),
        float(vals['harmonic_ratio_skewness']),
        float(vals['harmonic_ratio_kurtosis']),
        float(vals['harmonic_ratio_rms']),
        float(vals['harmonic_ratio_delta'])
        ))
        connection.commit()

    return harmonic_ratio_data

# Test print
#print(processed_data_odd_even_harmonic_ratio)
#endregion

"""
=====================================
|        High-Level Features          |
=====================================
"""

#region Danceability Extraction
def extract_danceability(track_id_to_path):

    danceability_data = {}

    for idx, (track_id, filepath) in enumerate(track_id_to_path.items(), start=1):
        print(f"[{idx}] [Danceability] Processing track {track_id} from {filepath}")

        extractor = es.MusicExtractor()
        features, _ = extractor(filepath)  
        #print(sorted(features.descriptorNames()))

        danceability = features['rhythm.danceability']
        danceability_data[track_id] = {
            'danceability': float(danceability)
        }
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, danceability) values (?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    danceability = excluded.danceability"""
        vals = danceability_data[track_id]     
        cursor.execute(sql, (
        track_id,
        danceability
        ))
        connection.commit()
        
    return danceability_data

# Test print
#print(processed_data_danceability)
#endregion

#region Dynamic Complexity Extraction
def extract_dynamic_complexity(track_id_to_path):
    dynamic_complexity_data = {}

    for idx, (track_id, filepath) in enumerate(track_id_to_path.items(), start=1):
        print(f"[{idx}] [Dynamic Complexity] Processing track {track_id} from {filepath}")

        extractor = es.MusicExtractor()
        features, _ = extractor(filepath)

        dynamic_complexity = features['lowlevel.dynamic_complexity']
        dynamic_complexity_data[track_id] = {
            'dynamic_complexity': float(dynamic_complexity)
        }
        cursor = connection.cursor()
        sql = """INSERT INTO processed_sound_data (
                    track_id, dynamic_complexity) values (?, ?) 
                    ON CONFLICT(track_id) DO UPDATE SET 
                    dynamic_complexity = excluded.dynamic_complexity"""
        vals = dynamic_complexity_data[track_id]     
        cursor.execute(sql, (
        track_id,
        dynamic_complexity
        ))
        connection.commit()
        
    return dynamic_complexity_data


# Test print
#print(processed_data_dynamic_complexity)
#endregion

#region Data Merge
def merge_audio_features(feature_dicts, metadata_df, on="track_id"):

    feature_dfs = []

    for idx, feature_dict in enumerate(feature_dicts):
        first_val = next(iter(feature_dict.values()))
        feature_prefix = f"feat_{idx}"

        if isinstance(first_val, dict):
            df = pd.DataFrame.from_dict(feature_dict, orient='index')
            df[on] = df.index.astype(int)
            df = df.rename(columns=lambda col: f"{feature_prefix}_{col}" if col != on else col)

        elif isinstance(first_val, (np.ndarray, list)):
            flattened_rows = []
            for track_id, val in feature_dict.items():
                arr = np.array(val)
                if arr.ndim == 1:
                    features = arr
                elif arr.ndim == 2:
                    features = np.mean(arr, axis=0)
                elif arr.ndim == 3:
                    features = np.mean(arr.reshape(arr.shape[0], -1), axis=0)
                else:
                    raise ValueError(f"Unsupported array shape {arr.shape} for track {track_id}")
                feature_row = {f"{feature_prefix}_f{i}": v for i, v in enumerate(features)}
                feature_row[on] = track_id
                flattened_rows.append(feature_row)
            df = pd.DataFrame(flattened_rows)

        elif isinstance(first_val, tuple):
            flattened_rows = []
            for track_id, (arr1, arr2) in feature_dict.items():
                mean1 = np.mean(arr1) if isinstance(arr1, (np.ndarray, list)) and len(arr1) > 0 else np.nan
                mean2 = np.mean(arr2) if isinstance(arr2, (np.ndarray, list)) and len(arr2) > 0 else np.nan
                flattened_rows.append({on: track_id, f"{feature_prefix}_mean1": mean1, f"{feature_prefix}_mean2": mean2})
            df = pd.DataFrame(flattened_rows)

        elif isinstance(first_val, (float, int)):
            df = pd.DataFrame([{on: tid, f"{feature_prefix}_value": val} for tid, val in feature_dict.items()])

        else:
            raise ValueError(f"Unsupported feature_dict format: {type(first_val)}")

        feature_dfs.append(df)

    merged_features = feature_dfs[0]
    for df in feature_dfs[1:]:
        merged_features = pd.merge(merged_features, df, on=on, how='outer')

    metadata_df[on] = metadata_df[on].astype(int)
    final_df = pd.merge(metadata_df, merged_features, on=on, how='inner')

    return final_df







#data.info()
#endregion



#region Process Load Audio Files

def process(track_id_to_path,mono_loaded_audio, eqloud_loaded_audio):
    #Call the function to extract pitch using Yin
    yin_processed_data_pitch = extract_pitch_yin(eqloud_loaded_audio)
    #Call the function to extract pitch using Melodia
    processed_data_pitch_melodia = extract_pitch_melodia(eqloud_loaded_audio)
    # Call the function to extract melodic pitch range
    processed_data_melodic_pitch_range = extract_melodic_pitch_range(eqloud_loaded_audio)
    # Call it on your Melodia output:
    processed_data_mnn = extract_mnn_stats(processed_data_pitch_melodia)
    # Call the function to extract inharmonicity
    processed_data_inharmonicity = extract_inharmonicity(mono_loaded_audio)
    # Call extract_chromagram
    processed_data_chromogram = extract_chromagram(mono_loaded_audio)
    # Call it
    processed_data_hpcp = extract_hpcp(mono_loaded_audio)
    # Call the function to extract key
    processed_data_key = extract_key(mono_loaded_audio)
    # Call the function to extract chord progression from HPCP
    processed_data_chord_progression = extract_chord_progression_from_hpcp(processed_data_hpcp)
    # Call the function to extract BPM
    processed_data_bpm = extract_bpm(mono_loaded_audio)
    # Call the function to extract spectral peaks
    processed_spectral_peaks = extract_spectral_peaks(mono_loaded_audio)
    # Call the function to extract onset rate
    processed_data_onset_rate = extract_onset_rate(mono_loaded_audio)
    #print(processed_data_onset_rate)
    # Call the function to extract loudness mean
    processed_loudness_mean = extract_loudness(mono_loaded_audio)
    # Call the function to extract RMS energy standard deviation
    processed_rms_energy_std = extract_rms_energy_std(mono_loaded_audio)
    # Call the function to extract dynamic range
    processed_dynamic_range = extract_dynamic_range(mono_loaded_audio)
    # Call the function to extract MFCCs
    processed_data_mfcc = extract_mfcc(mono_loaded_audio)
    # Call the function to extract spectral centroid
    processed_data_spectral_centroid = extract_spectral_centroid(mono_loaded_audio) #TODO: CHECK WHY 0S
    # Call the function to extract segment counts
    processed_segment_data = extract_segment_boundaries_and_novelty(eqloud_loaded_audio) 
    # Call the function to extract segment durations stats
    processed_data_segment_durations = extract_segment_durations_stats(processed_segment_data, eqloud_loaded_audio) 
    # Call the function to extract novelty stats
    processed_data_novelty_stats = extract_novelty_stats(processed_segment_data, eqloud_loaded_audio) 
    # Call the function to extract Log Attack Time
    processed_data_log_attack_time = extract_log_attack_time(mono_loaded_audio)
    # Call the function to extract spectral flatness
    processed_data_spectral_flatness = extract_spectral_flatness(mono_loaded_audio)
    #Call the function to extract vibrato presence
    #processed_data_vibrato = extract_vibrato(mono_loaded_audio) 
    # Call the function to extract Tristimulus coefficients
    processed_data_tristimulus = extract_tristimulus(mono_loaded_audio)
    # Call the function to extract Odd/Even Harmonic Energy Ratio
    processed_data_odd_even_harmonic_ratio = extract_odd_even_harmonic_ratio(mono_loaded_audio)
    # Call the function to extract danceability
    processed_data_danceability = extract_danceability(track_id_to_path)
    # Call the function to extract dynamic complexity
    processed_data_dynamic_complexity = extract_dynamic_complexity(track_id_to_path)
    # Call the function to extract dissonance from spectral peaks
    #processed_data_dissonance = extract_dissonance_from_peaks(processed_spectral_peaks) #TODO: remove comment, takes too long to run




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
            yin_skewness_pitch = scipy.stats.skew(valid_yin_pitches)
            yin_kurtosis_pitch = scipy.stats.kurtosis(valid_yin_pitches)
            yin_rms_pitch = np.sqrt(np.mean(valid_yin_pitches**2))
            yin_delta_pitch = np.mean(np.diff(valid_yin_pitches)) if len(valid_yin_pitches) > 1 else 0.0

            cursor = connection.cursor()
            sql = """INSERT INTO processed_sound_data (
                track_id, yin_pitch_mean, yin_pitch_median, yin_pitch_std, yin_pitch_skewness, yin_pitch_kurtosis, yin_pitch_rms, yin_pitch_delta) values (?, ?, ?, ?,?,?,?,?) 
                ON CONFLICT(track_id) DO UPDATE SET 
                yin_pitch_mean = excluded.yin_pitch_mean,
                yin_pitch_median = excluded.yin_pitch_median,
                yin_pitch_std = excluded.yin_pitch_std,
                yin_pitch_skewness = excluded.yin_pitch_skewness, 
                yin_pitch_kurtosis = excluded.yin_pitch_kurtosis , 
                yin_pitch_rms = excluded.yin_pitch_rms, 
                yin_pitch_delta = excluded.yin_pitch_delta"""
            cursor.executemany(sql, [(track_id, yin_mean_pitch, yin_median_pitch, yin_std_pitch, yin_skewness_pitch, yin_kurtosis_pitch, yin_rms_pitch, yin_delta_pitch)])
            connection.commit()

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




    return 
    return [
            yin_processed_data_pitch,
            processed_data_pitch_melodia,
            processed_data_melodic_pitch_range,
            processed_data_mnn,
            processed_data_inharmonicity,
            processed_data_chromogram,
            processed_data_hpcp,
            processed_data_key,
            processed_data_chord_progression,
            processed_spectral_peaks,
            processed_data_dissonance,
            processed_data_bpm,
            processed_data_onset_rate,
            # processed_data_beat_histogram,
            processed_loudness_mean,
            processed_dynamic_range,
            processed_rms_energy_std,
            processed_data_mfcc,
            processed_data_spectral_centroid,
            processed_segment_data,
            processed_data_segment_durations,
            processed_data_novelty_stats,
            processed_data_log_attack_time,
            #processed_data_vibrato,
            processed_data_spectral_flatness,
            processed_data_tristimulus,
            processed_data_odd_even_harmonic_ratio,
            processed_data_danceability,
            processed_data_dynamic_complexity
        ]




def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [convert_to_serializable(i) for i in obj]
    return obj

def batch_process_audio(track_id_to_path, page=30, max_songs=None, json_filename='processed_audio_features.json'):
    track_ids = list(track_id_to_path.keys())
    total_processed = 0

    fieldnames = [
        "yin_processed_data_pitch",
        "processed_data_pitch_melodia",
        "processed_data_melodic_pitch_range",
        "processed_data_mnn",
        "processed_data_inharmonicity",
        "processed_data_chromogram",
        "processed_data_hpcp",
        "processed_data_key",
        "processed_data_chord_progression",
        "processed_spectral_peaks",
        "processed_data_bpm",
        "processed_data_onset_rate",
        "processed_loudness_mean",
        "processed_dynamic_range",
        "processed_rms_energy_std",
        "processed_data_mfcc",
        "processed_data_spectral_centroid",
        "processed_segment_data",
        "processed_data_segment_durations",
        "processed_data_novelty_stats",
        "processed_data_log_attack_time",
        "processed_data_spectral_flatness",
        "processed_data_vibrato",
        "processed_data_tristimulus",
        "processed_data_odd_even_harmonic_ratio",
        "processed_data_danceability",
        "processed_data_dynamic_complexity"
    ]

 
        
    

    for batch_start in range(0, len(track_ids), page):
        if max_songs is not None and total_processed >= max_songs:
            break

        batch_track_ids = track_ids[batch_start : batch_start + page]
        batch_track_id_to_path = {tid: track_id_to_path[tid] for tid in batch_track_ids}

        mono_loaded_audio = {}
        eqloud_loaded_audio = {}
        batch_count = 0

        for track_id in batch_track_ids:
            if max_songs is not None and total_processed >= max_songs:
                break

            filepath = track_id_to_path[track_id]
            if not os.path.exists(filepath) or os.path.getsize(filepath) < 1024:
                continue

            try:
                mono_audio = es.MonoLoader(filename=filepath)()
                eqloud_audio = es.EqloudLoader(filename=filepath, sampleRate=44100)()
                mono_loaded_audio[track_id] = mono_audio
                eqloud_loaded_audio[track_id] = eqloud_audio
                batch_count += 1
                total_processed += 1
            except Exception:
                continue

        if mono_loaded_audio or eqloud_loaded_audio:
            print(f"Processing batch of {batch_count} tracks, batch starting at index {batch_start}")
            batch_results = process(batch_track_id_to_path, mono_loaded_audio, eqloud_loaded_audio)

            #print(batch_results)
            return 0
            '''if batch_results and isinstance(batch_results[0], (list, tuple)):
                processed_list = batch_results
            else:
                processed_list = [batch_results]

            for result in processed_list:
                item = {}
                for i, key in enumerate(fieldnames):
                    item[key] = convert_to_serializable(result[i]) if i < len(result) else None

                # Write comma before all but the first record to keep JSON valid
                if not first_record:
                    
                else:
                    first_record = False

                

    '''

    #print(f'Data written incrementally to {json_filename}')

batch_process_audio(track_id_to_path_another_name, page=10)  

#load mono audio files -> beat tracking, tempo estimation, onset detection, rhythmic analysis, uniform preprocessing


#endregion


