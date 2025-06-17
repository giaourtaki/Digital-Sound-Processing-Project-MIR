# https://www.youtube.com/watch?v=SW0YGA9d8y8
# https://github.com/microsoft/pylance-release/blob/main/TROUBLESHOOTING.md#unresolved-import-warnings
# source ./soundvenv/bin/activate
# to commit -> move pain.py to rep folder

#region Imports
import os


import IPython.display as ipd
import pandas as pd
import numpy as np
from numpy import pad

import librosa
#import librosa.display
#import essentia 
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
    df.drop(columns=['album_id', 'album_title', 'album_url', 'track_file', 'artist_id', 'artist_name', 'artist_url', 'artist_website', 'license_image_file', 'license_image_file_large', 'license_parent_id', 'license_title', 'license_url', 'tags', 'track_bit_rate', 'track_comments', 'track_composer', 'track_copyright_c', 'track_copyright_p', 'track_date_created', 'track_date_recorded', 'track_disc_number', 'track_duration', 'track_explicit', 'track_explicit_notes', 'track_favorites', 'track_image_file', 'track_information', 'track_instrumental', 'track_interest', 'track_language_code', 'track_listens', 'track_lyricist', 'track_number', 'track_publisher', 'track_title', 'track_url'], inplace=True)

    # remove empty genre columns
    df.dropna(subset=['track_genres'], inplace=True)
    
    
    return df

preprocess_data(data)
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

# Check if track is specific genre (e.g., Pop)
# Define genre ID mapping
genre_ids = {
    '10': 'Pop',
    '12': 'Rock',
    '31': 'Metal',
    '25': 'Punk'
}

# Set limits for each genre TODO: change this to 80 songs per genre
genre_limits = {
    'Pop': 80,
    'Rock': 80,
    'Metal': 80,
    'Punk': 80
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


# Build path map from matched track_ids TODO: fix, i need all 5 songs for each genre, not just the first 5
track_id_to_path = {}

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

track_id_to_path = collect_track_paths(base_dir, track_id_to_genre)
# Test print
print(f"Collected {len(track_id_to_path)} tracks:")
#missing_ids = set(track_id_to_genre.keys()) - set(track_id_to_path.keys())
#print("Missing track IDs (no .mp3 file found):", missing_ids)
#for tid, path in track_id_to_path.items():
  #  print(f"Track ID: {tid}, Genre: {track_id_to_genre[tid]}, Path: {path}")
#print(len(track_id_to_path))
#print(track_id_to_path)

#endregion

#region Load Audio Files


def process():
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
    #Call the function to extract pitch using Yin
    yin_processed_data_pitch = extract_pitch_yin(eqloud_loaded_audio)
    #Call the function to extract pitch using Melodia
    processed_data_pitch_melodia = extract_pitch_melodia(eqloud_loaded_audio)
    # Call the function to extract melodic pitch range
    processed_data_melodic_pitch_range = extract_melodic_pitch_range(eqloud_loaded_audio)
    return 0


#load mono audio files -> beat tracking, tempo estimation, onset detection, rhythmic analysis, uniform preprocessing

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

            # Log progress every 10 seconds of audio TODO: this doesnt work
            if i % (sample_rate * 10) == 0 and i != 0:
                print(f"    ↳ Processed {i / sample_rate:.1f}s")

        yin_processed_data_pitch[track_id] = {
            'yin_pitch_values': pitches,
            'yin_pitch_confidence': confidences,
            'yin_pitch_times': times
        }

    return yin_processed_data_pitch

#pitch test print
#print(yin_processed_data_pitch)


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

#region Melody / Predominant pitch Extraction -> Estimates the fundamental frequency of the predominant melody from polyphonic music signals using the MELODIA algorithm
def extract_pitch_melodia(eqloud_loaded_audio, frame_size=2048, hop_size=128, sample_rate=44100): #TODO: justify the hop size

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
                'melodia_pitch_times': melodia_pitch_times
            }

        except Exception as e:
            print(f"Error processing track {track_id}: {e}")
            continue

    return processed_data_pitch_melodia



#Test print
#print(processed_data_pitch_melodia)
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

    return melodic_pitch_range_data

# Test print
#print(processed_data_melodic_pitch_range)
#endregion

#region MIDI Note Number (MNN) Statistics #TODO: check if this is correct
def extract_mnn_stats(processed_data_pitch_melodia):
    """
    Given a dict
      { track_id: {
          'melodia_pitch_values': np.array([...Hz...]),
          'melodia_pitch_confidence': np.array([...]),
          'melodia_pitch_times': np.array([...])
        }, ... }
    convert each non-zero Hz pitch into MIDI note numbers,
    round to the nearest semitone, and compute mean/median/std.
    Returns a dict:
      { track_id: {'mnn_mean':…, 'mnn_median':…, 'mnn_std':…}, … }
    """
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
                'mnn_std':    np.std(mnn_int)
            }
        else:
            # No valid pitches detected
            processed_data_mnn[track_id] = {
                'mnn_mean':   np.nan,
                'mnn_median': np.nan,
                'mnn_std':    np.nan
            }

    return processed_data_mnn

# Call it on your Melodia output:
processed_data_mnn = extract_mnn_stats(processed_data_pitch_melodia)

# Example: turn into a DataFrame and merge back into `data`
mnn_features = [
    dict(track_id=tid, **stats)
    for tid, stats in processed_data_mnn.items()
]
mnn_df = pd.DataFrame(mnn_features)

# Keep only processed tracks
data = data[data['track_id'].isin(mnn_df['track_id'])]

# Merge
data = data.merge(mnn_df, on='track_id', how='left')

# Test print
#print(data[['track_id', 'mnn_mean', 'mnn_median', 'mnn_std']].head(10))
#print(data.info())
#print(data)
#endregion

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
            'inharmonicity_std': std_inharmonicity
        }

    return processed_data_inharmonicity
# Call the function to extract inharmonicity
processed_data_inharmonicity = extract_inharmonicity(mono_loaded_audio)
# Test print
#print(processed_data_inharmonicity)
#endregion

#region Chromagram extraction #TODO: check if this works/name variables better


def extract_chromagram(mono_loaded_audio,
                       frame_size=2048,
                       hop_size=1024,
                       sample_rate=44100):
    processed_data_chromogram = {}

    window   = es.Windowing(type='hann',   size=frame_size)
    spectrum = es.Spectrum()
    chroma   = es.Chromagram()  # expects CQT‐length spectrum

    total = len(mono_loaded_audio)
    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total}] [Chroma] Processing track {track_id}")

        frames = []
        for i in range(0, len(audio)-frame_size, hop_size):
            frame = window(audio[i:i+frame_size])
            spec  = spectrum(frame)            # length = 1025
            # pad to 32768
            spec_padded = pad(spec, (0, 32768 - spec.shape[0]), mode='constant')
            c = chroma(spec_padded)            
            frames.append(c)

        processed_data_chromogram[track_id] = np.vstack(frames) if frames else np.empty((0,12))

    return processed_data_chromogram


# Call it
processed_data_chromogram = extract_chromagram(mono_loaded_audio)
# Test print
#print(processed_data_chromogram)

#endregion

#region HPCP Extraction

def extract_hpcp(mono_loaded_audio,
                 frame_size=2048,
                 hop_size=1024,
                 sample_rate=44100,
                 bins_per_octave=12,
                 min_freq=50,
                 max_freq=5000):
    """
    Returns { track_id: np.array(n_frames,12) } using HPCP (a chromagram).
    """
    processed_data_hpcp = {}

    window       = es.Windowing(type='hann', size=frame_size)
    spectrum     = es.Spectrum()
    spectral_peaks = es.SpectralPeaks(orderBy='magnitude', magnitudeThreshold=0.01)
    hpcp         = es.HPCP(size=bins_per_octave,
                           minFrequency=min_freq,
                           maxFrequency=max_freq,
                           referenceFrequency=440.0)

    total = len(mono_loaded_audio)
    for idx, (tid, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total}] [HPCP] Processing track {tid}")
        frames = []
        for i in range(0, len(audio)-frame_size, hop_size):
            frame = window(audio[i:i+frame_size])
            spec  = spectrum(frame)
            freqs, mags = spectral_peaks(spec)
            c = hpcp(freqs, mags)  # yields 12-bin chroma
            frames.append(c)

        processed_data_hpcp[tid] = np.vstack(frames) if frames else np.empty((0, bins_per_octave))

    return processed_data_hpcp

# Call it
processed_data_hpcp = extract_hpcp(mono_loaded_audio)
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

    return processed_data_key

# Call the function to extract key
processed_data_key = extract_key(mono_loaded_audio)
# Test print
#print(processed_data_key)
#endregion

#region Chord Extraction
def extract_chord_progression_from_hpcp(processed_data_hpcp):
    """
    Extracts chord progression from precomputed HPCPs using Essentia's ChordsDetection algorithm.

    Parameters:
        processed_data_hpcp (dict): Dictionary mapping track IDs to numpy arrays of HPCP vectors (shape: n_frames x 12).

    Returns:
        dict: Dictionary mapping track IDs to lists of detected chord labels.
    """

    import essentia.standard as es

    chord_progression_data = {}
    chords_detector = es.ChordsDetection()

    for idx, (track_id, hpcp_array) in enumerate(processed_data_hpcp.items(), start=1):
        print(f"[{idx}/{len(processed_data_hpcp)}] [Chord Progression] Processing HPCP for track {track_id}")

        try:
            # Convert numpy array to list of lists if necessary
            if hasattr(hpcp_array, 'tolist'):
                hpcps = hpcp_array.tolist()
            else:
                hpcps = hpcp_array

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

    return chord_progression_data

# Call the function to extract chord progression from HPCP
processed_data_chord_progression = extract_chord_progression_from_hpcp(processed_data_hpcp)
# Test print
#print(processed_data_chord_progression)
#endregion

#region Spectral Peaks Extraction
def extract_spectral_peaks(mono_loaded_audio, frame_size=2048, hop_size=1024):
    """
    Extract average spectral peaks per track.
    Returns {track_id: (sorted_avg_frequencies, sorted_avg_magnitudes)}
    """
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

    return peak_data
# Call the function to extract spectral peaks
processed_spectral_peaks = extract_spectral_peaks(mono_loaded_audio)
# Test print
#print(processed_spectral_peaks)
# Shorting Frequencies and Magnitudes Together for Dissonance extracion


#end region
#region Dissonance Extraction

def extract_dissonance_from_peaks(processed_spectral_peaks):
    """
    Calculates dissonance from spectral peaks using Essentia's Dissonance algorithm.

    Parameters:
        processed_spectral_peaks (dict): Dictionary mapping track IDs to a tuple (frequencies, magnitudes),
                                         where both are lists or numpy arrays sorted by frequency.

    Returns:
        dict: Dictionary mapping track IDs to scalar dissonance values.
    """

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

    return dissonance_data
# Call the function to extract dissonance from spectral peaks
#processed_data_dissonance = extract_dissonance_from_peaks(processed_spectral_peaks) #TODO: remove comment, takes too long to run
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
    """
    Extracts BPM and standard deviation of instantaneous tempo from audio tracks.

    Parameters:
        mono_loaded_audio (dict): Mapping of track IDs to mono audio arrays.
        sample_rate (int): Sampling rate of the audio (default 44100 Hz).

    Returns:
        dict: Mapping track IDs to {'bpm': float, 'ticks': list, 'tempo_std': float}
    """
    
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
# Call the function to extract BPM
processed_data_bpm = extract_bpm(mono_loaded_audio)
# Test print
#print(processed_data_bpm)
#endregion

#region Onset Extraction
def extract_onset_rate(mono_loaded_audio, sample_rate=44100):
    """
    Extract onset rate (number of onsets per second) from mono audio tracks.
    Returns a dictionary with onset rate per track.
    """
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
# Call the function to extract onset rate

processed_data_onset_rate = extract_onset_rate(mono_loaded_audio)
# Test print
#print(processed_data_onset_rate)
#endregion

#region Beat Histogram Extraction TODO: fucking fix this
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

#processed_data_beat_histogram = extract_beat_histogram(mono_loaded_audio) TODO: fucking fix this
#Test print
#print(processed_data_beat_histogram)
#endregion

"""
=====================================
|        Dynamic Features          |
=====================================
"""
#region Loudness Mean Extraction
def extract_loudness_mean(mono_loaded_audio, sample_rate=44100, frame_size=2048, hop_size=1024):
    """
    Extract mean loudness per track using Essentia Loudness algorithm.
    Returns dict {track_id: mean loudness}
    """
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

            mean_loudness = np.mean(loudness_values) if loudness_values else None
            processed_loudness[track_id] = mean_loudness

        except Exception as e:
            print(f"Error processing loudness for track {track_id}: {e}")
            processed_loudness[track_id] = None

    return processed_loudness
# Call the function to extract loudness mean
processed_loudness_mean = extract_loudness_mean(mono_loaded_audio)
# Test print
#print(processed_loudness_mean)
#endregion

#region Dynamic Range Extraction
def extract_dynamic_range(mono_loaded_audio, frame_size=2048, hop_size=1024):
    """
    Extract dynamic range (max RMS - min RMS) per track.
    Returns dict {track_id: dynamic_range}
    """
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
                dynamic_range = max(rms_values) - min(rms_values)
            else:
                dynamic_range = None

            processed_dynamic_range[track_id] = dynamic_range

        except Exception as e:
            print(f"Error processing dynamic range for track {track_id}: {e}")
            processed_dynamic_range[track_id] = None

    return processed_dynamic_range
# Call the function to extract dynamic range
processed_dynamic_range = extract_dynamic_range(mono_loaded_audio)
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

            rms_std = np.std(rms_values) if rms_values else None
            processed_rms_std[track_id] = rms_std

        except Exception as e:
            print(f"Error processing RMS std for track {track_id}: {e}")
            processed_rms_std[track_id] = None

    return processed_rms_std
# Call the function to extract RMS energy standard deviation
processed_rms_energy_std = extract_rms_energy_std(mono_loaded_audio)
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
    """
    Extract MFCCs from raw audio using Essentia.
    Input: dict {track_id: mono audio (VectorReal)}
    Output: dict {track_id: {mfcc_coefficients, mfcc_bands, mfcc_times, mfcc_mean, mfcc_std}}
    """
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
            mean = mfcc_array.mean(axis=0).tolist()
            std = mfcc_array.std(axis=0).tolist()
        else:
            mean = []
            std = []

        processed_data_mfcc[track_id] = {
            'mfcc_coefficients': mfcc_coeffs,
            'mfcc_bands': mfcc_bands,
            'mfcc_times': times,
            'mfcc_mean': mean,
            'mfcc_std': std
        }

    return processed_data_mfcc

# Call the function to extract MFCCs
processed_data_mfcc = extract_mfcc(mono_loaded_audio)
# Test print
#print(processed_data_mfcc)
#endregion

#region Spectral Centroid Extraction
def extract_spectral_centroid(mono_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):
    """
    Extract spectral centroid from preloaded mono audio.
    Input: dict {track_id: mono audio (VectorReal)}
    Output: dict {track_id: {centroid_values, times, centroid_mean, centroid_std}}
    """
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
            mean = centroid_array.mean().item()
            std = centroid_array.std().item()
        else:
            mean = 0.0
            std = 0.0

        centroid_processed_data[track_id] = {
            'centroid_values': centroids,
            'centroid_times': times,
            'centroid_mean': mean,
            'centroid_std': std
        }

    return centroid_processed_data
# Call the function to extract spectral centroid
processed_data_spectral_centroid = extract_spectral_centroid(mono_loaded_audio) #TODO: CHECK WHY 0S
# Test print
#print(processed_data_spectral_centroid)

#endregion
"""
=====================================
|        Form Features          |
=====================================
"""
#region Segment Count Extraction
from scipy.signal import find_peaks
def extract_segment_boundaries_and_novelty(eqloud_loaded_audio, frame_size=1024, hop_size=512, sample_rate=44100):
    """
    Returns per track:
    - segment boundaries (seconds)
    - onset envelope (novelty curve)
    - audio duration (seconds)
    """
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
# Call the function to extract segment counts
processed_segment_data = extract_segment_boundaries_and_novelty(eqloud_loaded_audio) #TODO: logika thelei mono
# Test print
#print(processed_data_segment_count)
#endregion

#region Segment Duration Extraction
def extract_segment_durations_stats(segment_data):
    """
    Input:
        segment_data: dict from extract_segment_boundaries_and_novelty
    Returns:
        dict per track:
          - segment_durations (list of floats in seconds)
          - mean_duration (float)
          - std_duration (float)
    """
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

        durations_stats[track_id] = {
            'segment_durations': durations.tolist(),
            'mean_duration': float(np.mean(durations)),
            'std_duration': float(np.std(durations))
        }

    return durations_stats
# Call the function to extract segment durations stats
processed_data_segment_durations = extract_segment_durations_stats(processed_segment_data)
# Test print
#print(processed_data_segment_durations)
#endregion

#region Novelty Curve Extraction
def extract_novelty_stats(segment_data): #TODO: logika thelei mono
    """
    Input:
      segment_data: dict from extract_segment_boundaries_and_novelty
    Returns:
      dict per track:
        - onset_envelope (novelty curve) list of floats
        - mean_onset (float)
        - std_onset (float)
    """
    novelty_stats = {}
    total_tracks_len = len(eqloud_loaded_audio)
    for idx, (track_id, data) in enumerate(segment_data.items(), start=1):
        print(f"[{idx}/{total_tracks_len}] [Novelty Curve] Processing track {track_id}")
        onset_env = data['onset_envelope']
        novelty_stats[track_id] = {
            'onset_envelope': onset_env.tolist(),
            'mean_onset': float(np.mean(onset_env)),
            'std_onset': float(np.std(onset_env))
        }

    return novelty_stats
# Call the function to extract novelty stats
processed_data_novelty_stats = extract_novelty_stats(processed_segment_data)
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
    """
    Extract Log Attack Time (LAT) per track using Essentia's LogAttackTime algorithm.

    Parameters:
        mono_loaded_audio (dict): Dictionary mapping track IDs to loaded audio arrays.
        frame_size (int): Frame size in samples for analysis.
        hop_size (int): Hop size in samples between frames.
        sample_rate (int): Sampling rate of the audio.

    Returns:
        dict: Mapping track IDs to extracted Log Attack Time values per frame.
    """

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

        lat_processed_data[track_id] = {
            'log_attack_time_values': lat_values,
            'log_attack_time_times': times
        }

    return lat_processed_data
# Call the function to extract Log Attack Time
processed_data_log_attack_time = extract_log_attack_time(mono_loaded_audio)
# Test print
#print(processed_data_log_attack_time)
#endregion

#region Vibrato Extraction
def extract_vibrato(mono_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):
    """
    Extract Vibrato Presence per track using Essentia's VibratoPresence algorithm.

    Parameters:
        mono_loaded_audio (dict): Dictionary mapping track IDs to loaded audio arrays.
        frame_size (int): Frame size in samples for analysis.
        hop_size (int): Hop size in samples between frames.
        sample_rate (int): Sampling rate of the audio.

    Returns:
        dict: Mapping track IDs to extracted Vibrato  values per frame.
    """

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

        vibrato_data[track_id] = {
            'vibrato_values': vibrato_values,
            'vibrato_times': times
        }

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
    """
    Compute average spectral flatness for each track.

    Returns:
        dict: Mapping track IDs to average spectral flatness.
    """
    flatness_data = {}

    total_tracks = len(mono_loaded_audio)
    for idx, (track_id, audio) in enumerate(mono_loaded_audio.items(), start=1):
        print(f"[{idx}/{total_tracks}] [Spectral Flatness] Processing track {track_id}")

        try:
            # Convert to numpy array (Librosa expects float32)
            audio_np = np.array(audio).astype(np.float32)
            S_flat = librosa.feature.spectral_flatness(y=audio_np, n_fft=frame_size, hop_length=hop_size)
            avg_flatness = float(np.mean(S_flat))

        except Exception as e:
            print(f"Error processing spectral flatness for track {track_id}: {e}")
            avg_flatness = None

        flatness_data[track_id] = {
            'avg_spectral_flatness': avg_flatness
        }

    return flatness_data
# Call the function to extract spectral flatness
processed_data_spectral_flatness = extract_spectral_flatness(mono_loaded_audio)
# Test print
#print(processed_data_spectral_flatness)
#endregion


#region Tristimulus Extraction
def extract_tristimulus(mono_loaded_audio, frame_size=2048, hop_size=1024, sample_rate=44100):
    """
    Extract average Tristimulus coefficients (T1, T2, T3) per track.

    Returns:
        dict: Mapping track IDs to average T1, T2, T3.
    """
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
            tristimulus_data[track_id] = {
                'tristimulus_t1': float(np.mean(t1_vals)),
                'tristimulus_t2': float(np.mean(t2_vals)),
                'tristimulus_t3': float(np.mean(t3_vals))
            }
        else:
            tristimulus_data[track_id] = {
                'tristimulus_t1': None,
                'tristimulus_t2': None,
                'tristimulus_t3': None
            }

    return tristimulus_data
# Call the function to extract Tristimulus coefficients
processed_data_tristimulus = extract_tristimulus(mono_loaded_audio)
# Test print
#print(processed_data_tristimulus)
#endregion


#region Odd/Even Harmonic Energy Ratio Extraction
def extract_odd_even_harmonic_ratio(mono_loaded_audio, frame_size=2048, hop_size=1024):
    """
    Extract average Odd-to-Even Harmonic Energy Ratio per track.

    Returns:
        dict: Mapping track IDs to average Odd/Even ratio.
    """
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

        harmonic_ratio_data[track_id] = {
            'odd_even_harmonic_ratio': float(np.mean(ratios)) if ratios else None
        }

    return harmonic_ratio_data
# Call the function to extract Odd/Even Harmonic Energy Ratio
processed_data_odd_even_harmonic_ratio = extract_odd_even_harmonic_ratio(mono_loaded_audio)
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

    return danceability_data
# Call the function to extract danceability
processed_data_danceability = extract_danceability(track_id_to_path)
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

    return dynamic_complexity_data

# Call the function to extract dynamic complexity
processed_data_dynamic_complexity = extract_dynamic_complexity(track_id_to_path)
# Test print
#print(processed_data_dynamic_complexity)
#endregion

#region Data Merge
def merge_audio_features(feature_dicts, metadata_df, on="track_id"):
    """
    Merges multiple feature dictionaries and joins with metadata dataframe.

    Parameters:
    - feature_dicts (list): List of dicts {track_id: features}, where features can be:
        - dict of named features
        - np.ndarray or list (1D or 2D)
        - scalar (float, int)
        - tuple of (np.ndarray, np.ndarray)
    - metadata_df (pd.DataFrame): DataFrame containing metadata with 'track_id'.
    - on (str): Column to merge on.

    Returns:
    - pd.DataFrame: Combined DataFrame with features and metadata.
    """
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



data = merge_audio_features(
    feature_dicts=[
        processed_data_pitch_melodia,
        processed_data_melodic_pitch_range,
        processed_data_mnn,
        processed_data_inharmonicity,
        processed_data_chromogram,
        processed_data_hpcp,
        processed_data_key,
        processed_data_chord_progression,
        processed_spectral_peaks,
        #processed_data_dissonance,
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
    ],
    metadata_df=data
)

#Test print
#data.to_csv('processed_audio_features.csv', index=False)
#data.info()
#endregion









# Create Features / Target Variables (Make flashcards)



# ML Processing



# Hyperparameter Tuning



# Predictions and Evaluation



# Plot

